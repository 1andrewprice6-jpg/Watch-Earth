import React, { useRef, useMemo, Suspense, useEffect } from 'react';
import { Canvas, useFrame, useThree, extend } from '@react-three/fiber';
import { OrbitControls, Html, Billboard, useTexture, DeviceOrientationControls, Effects } from '@react-three/drei';
import { EffectComposer, Bloom, ChromaticAberration, Noise } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import { TAARenderPass, UnrealBloomPass, ShaderPass } from 'three-stdlib';
import * as THREE from 'three';

extend({ TAARenderPass, UnrealBloomPass, ShaderPass });

const bloomVertexShader = `
  varying vec2 vUv;
  varying vec3 vWorldPosition;
  void main() {
    vUv = uv;
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const bloomFragmentShader = `
  varying vec2 vUv;
  varying vec3 vWorldPosition;
  uniform vec3 uColor;
  uniform float uOpacity;
  
  void main() {
    vec2 centered = vUv - 0.5;
    
    // Perspective Scalar Calibration (Angle of Exposure)
    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
    float incidenceAngle = max(0.0, viewDir.y);
    float horizonFactor = 1.0 - incidenceAngle;
    
    // Atmospheric Compression (squash vertically, spread horizontally at horizon)
    vec2 distortedUV = centered;
    distortedUV.y *= 1.0 + (horizonFactor * 1.5);
    distortedUV.x *= 1.0 - (horizonFactor * 0.3);
    
    float dist = length(distortedUV) * 2.0;
    
    // Angle of Exposure logic (Summer-Winter Effect)
    float distFromCenter = length(vWorldPosition.xz);
    // Maps position: Tropic of Cancer ~ 1.25 (Summer), Tropic of Capricorn ~ 3.75 (Winter)
    float seasonFactor = clamp((distFromCenter - 1.25) / 2.5, 0.0, 1.0);
    
    // In winter, the sun is farther and lower in the sky (horizonFactor increases).
    // The "Angle of Exposure" reduces glow intensity and reddens the color.
    float atmosphericDensity = mix(seasonFactor, 1.0, horizonFactor * 0.8);
    
    // Smooth falloff from center to edge (tightens near horizon and in winter)
    float alpha = pow(smoothstep(1.0, 0.0, dist), 2.0 + horizonFactor * 1.0 + atmosphericDensity) * uOpacity;
    
    // Hot core
    float core = pow(smoothstep(0.4, 0.0, dist), 3.0);
    alpha += core * 0.5 * uOpacity;
    
    if (alpha <= 0.01) discard;
    
    // Shift base color to warmer/dimmer tones in winter/low angles
    vec3 finalColor = mix(uColor, vec3(1.0, 0.5, 0.1), atmosphericDensity * 0.8);
    
    // Simulate atmospheric red-shifting near the horizon
    float redshift = smoothstep(0.4, 1.0, horizonFactor);
    finalColor.r += redshift * 0.4 * atmosphericDensity;
    finalColor.g -= redshift * 0.2 * atmosphericDensity;
    finalColor.b -= redshift * 0.5 * atmosphericDensity;
    
    // Attenuate opacity based on density (winter sun looks less intense, more diffuse)
    alpha *= mix(1.0, 0.4, atmosphericDensity * horizonFactor);
    
    gl_FragColor = vec4(finalColor, alpha);
  }
`;

const sunVertexShader = `
  varying vec2 vUv;
  varying vec3 vWorldPosition;
  void main() {
    vUv = uv;
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    vec3 viewDir = normalize(cameraPosition - worldPos.xyz);
    float incidenceAngle = max(0.0, viewDir.y);
    float horizonFactor = 1.0 - incidenceAngle;
    
    // Perspective Compression: flatten the sphere when viewed near the horizon
    worldPos.y -= (position.y * horizonFactor * 0.4);
    
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const sunFragmentShader = `
  uniform float uTime;
  uniform vec3 uColor;
  varying vec2 vUv;
  varying vec3 vWorldPosition;

  // 3D Simplex noise functions
  vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
  float snoise(vec3 v){
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(
               i.z + vec4(0.0, i1.z, i2.z, 1.0))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0))
             + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = inversesqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  void main() {
    float n1 = snoise(vec3(vUv * 8.0, uTime * 0.4));
    float n2 = snoise(vec3(vUv * 16.0, uTime * 0.8));
    float noiseValue = (n1 + n2 * 0.5) * 0.5 + 0.5;
    
    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
    float incidenceAngle = max(0.0, viewDir.y);
    float horizonFactor = 1.0 - incidenceAngle;

    // Angle of Exposure logic (Summer-Winter Effect)
    float distFromCenter = length(vWorldPosition.xz);
    float seasonFactor = clamp((distFromCenter - 1.25) / 2.5, 0.0, 1.0);
    
    // Closer to horizon or outer ring (winter) -> thicker atmospheric density
    float atmosphericDensity = mix(seasonFactor, 1.0, horizonFactor * 0.9);
    
    vec3 brightColor = uColor + vec3(0.5, 0.4, 0.2); // Core bright spots
    vec3 darkColor = uColor * 0.6; // Darker regions
    
    // Shift base color to warmer/dimmer tones in winter/low angles
    brightColor = mix(brightColor, vec3(1.0, 0.5, 0.1), atmosphericDensity * 0.9);
    darkColor = mix(darkColor, vec3(0.6, 0.2, 0.05), atmosphericDensity * 0.9);
    
    vec3 finalColor = mix(darkColor, brightColor, noiseValue);

    // Make the Sun a "Black Sun" (Yin Dot in the Yang Light)
    // We will use the viewDir and normal to find the edges
    // Actually we don't have normals here, but we can use the UV coordinates
    vec2 center = vUv - 0.5;
    float dist = length(center) * 2.0;
    
    // Smooth transition from pitch black to glowing rim
    float rimLight = smoothstep(0.4, 0.95, dist);
    
    // The core is black
    vec3 blackCore = vec3(0.01, 0.01, 0.02);
    
    finalColor = mix(blackCore, finalColor * 2.0, rimLight);

    // Intensity drops slightly as it gets farther and lower
    float intensityBoost = 1.0 - min(atmosphericDensity * horizonFactor, 0.8);
    finalColor *= mix(1.0, 0.5, atmosphericDensity * horizonFactor) * intensityBoost;

    // Apply Perspective Scalar Redshift
    float redshift = smoothstep(0.4, 1.0, horizonFactor);
    finalColor.r += redshift * 0.5 * atmosphericDensity;
    finalColor.g -= redshift * 0.3 * atmosphericDensity;
    finalColor.b -= redshift * 0.6 * atmosphericDensity;
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

const starVertexShader = `
  uniform float uTime;
  uniform vec3 uSunPos;
  attribute float aScale;
  attribute float aPhase;
  varying float vAlpha;
  
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    gl_PointSize = aScale * (30.0 / -mvPosition.z);
    
    // Twinkle: slow sine wave interpolation
    float baseAlpha = 0.5 + 0.5 * sin(uTime * 1.5 + aPhase);
    
    // Fade stars near the sun for a realistic atmosphere
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vec3 viewDir = normalize(worldPosition.xyz);
    vec3 sunDir = normalize(uSunPos);
    float sunScattering = max(dot(viewDir, sunDir), 0.0);
    float fade = 1.0 - pow(sunScattering, 4.0); 
    
    vAlpha = baseAlpha * clamp(fade, 0.0, 1.0);
  }
`;

const starFragmentShader = `
  varying float vAlpha;
  
  void main() {
    float r = distance(gl_PointCoord, vec2(0.5));
    if (r > 0.5) discard;
    float glow = smoothstep(0.5, 0.0, r);
    gl_FragColor = vec4(vec3(1.0, 0.98, 0.95) * glow, vAlpha * glow);
  }
`;

function DynamicStarfield({ sunPosRef }) {
  const pointsRef = useRef<THREE.Points>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  const [positions, scales, phases] = useMemo(() => {
    const STAR_COUNT = 2000;
    const pos = new Float32Array(STAR_COUNT * 3);
    const scale = new Float32Array(STAR_COUNT);
    const phase = new Float32Array(STAR_COUNT);
    
    for (let i = 0; i < STAR_COUNT; i++) {
        const r = 20 + Math.random() * 30; // Between 20 and 50 units away
        const theta = 2 * Math.PI * Math.random();
        const phi = Math.acos(2 * Math.random() - 1);
        pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
        pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        pos[i * 3 + 2] = r * Math.cos(phi);

        scale[i] = 0.5 + Math.random() * 2.5; 
        phase[i] = Math.random() * Math.PI * 2; 
    }
    return [pos, scale, phase];
  }, []);

  useFrame((state) => {
     if (materialRef.current) {
        materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
        if (sunPosRef && sunPosRef.current) {
          materialRef.current.uniforms.uSunPos.value.copy(sunPosRef.current);
        }
     }
     if (pointsRef.current) {
        pointsRef.current.rotation.y = state.clock.elapsedTime * 0.005; // Slow rotation
     }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={positions.length / 3} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-aScale" count={scales.length} array={scales} itemSize={1} />
        <bufferAttribute attach="attributes-aPhase" count={phases.length} array={phases} itemSize={1} />
      </bufferGeometry>
      <shaderMaterial 
        ref={materialRef}
        vertexShader={starVertexShader}
        fragmentShader={starFragmentShader}
        uniforms={useMemo(() => ({ 
            uTime: { value: 0 },
            uSunPos: { value: new THREE.Vector3() } 
        }), [])}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

const skyVertexShader = `
  varying vec3 vWorldPosition;
  void main() {
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const skyFragmentShader = `
  varying vec3 vWorldPosition;
  uniform vec3 uSunPos;
  uniform vec3 uSunColor;
  
  void main() {
    vec3 viewDir = normalize(vWorldPosition - cameraPosition);
    vec3 sunDir = normalize(uSunPos - cameraPosition);
    
    // Angle between view direction and sun direction
    float cosTheta = dot(viewDir, sunDir);
    
    // Rayleigh Scattering (Blue Sky)
    float rayleighPhase = 0.75 * (1.0 + cosTheta * cosTheta);
    
    // Mie Scattering (Halo around sun)
    float g = 0.85; 
    float g2 = g * g;
    float miePhase = 1.5 * ((1.0 - g2) / (2.0 + g2)) * (1.0 + cosTheta * cosTheta) / pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
    
    // Atmosphere thickness based on view altitude (thicker near horizon)
    float viewAltitude = max(viewDir.y, 0.0);
    float opticalDepth = exp(-viewAltitude * 3.0); 
    
    vec3 rayleighColor = vec3(0.05, 0.15, 0.5) * rayleighPhase * opticalDepth * 2.5;
    vec3 mieColor = uSunColor * miePhase * 0.015;
    
    vec3 baseSky = vec3(0.002, 0.002, 0.008); 
    vec3 finalColor = baseSky + rayleighColor + mieColor;
    
    // Horizon representation
    // Sharp line fading out quickly
    float horizonLine = smoothstep(0.005, 0.0, abs(viewDir.y));
    float horizonGlow = smoothstep(0.1, 0.0, abs(viewDir.y));
    
    // More pronounced when the sun is lower 
    // uSunPos is the sun world position. sunDir is relative to camera, but the map is at y=0
    vec3 globalSunDir = normalize(uSunPos);
    float sunAltitude = max(globalSunDir.y, 0.0);
    
    // Exponential increase in horizon fog/glow as sun gets lower
    float horizonIntensity = exp(-sunAltitude * 10.0);
    
    // Color of the horizon blends sun color with an atmospheric haze color
    vec3 horizonHaze = mix(uSunColor, vec3(0.8, 0.9, 1.0), 0.3) * horizonIntensity;
    
    finalColor += horizonHaze * (horizonLine * 0.5 + horizonGlow * 0.8);
    
    // Tone mapping / exposure
    finalColor = 1.0 - exp(-finalColor * 1.5);
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

function AtmospherePolish() {
  const { camera } = useThree();
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  useFrame(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.viewVector.value.copy(camera.position);
    }
  });

  return (
    <mesh position={[0, 0, 0]}>
      {/* 5.05 to act as a firmament dome over the 5.0 radius earth disk */}
      <sphereGeometry args={[5.05, 64, 64]} />
      <shaderMaterial
        ref={materialRef}
        transparent={true}
        side={THREE.BackSide}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        uniforms={useMemo(() => ({
            glowColor: { value: new THREE.Color(0x3366ff) },
            viewVector: { value: new THREE.Vector3() }
        }), [])}
        vertexShader={`
            varying vec3 vNormal;
            varying vec3 vViewPosition;
            void main() {
                vNormal = normalize(normalMatrix * normal);
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                vViewPosition = -mvPosition.xyz;
                gl_Position = projectionMatrix * mvPosition;
            }
        `}
        fragmentShader={`
            varying vec3 vNormal;
            varying vec3 vViewPosition;
            uniform vec3 glowColor;
            void main() {
                // Fresnel intensity
                float intensity = pow(0.7 - dot(vNormal, normalize(vViewPosition)), 4.0);
                // Mask the bottom half so it only acts as a dome above y=0
                float mask = smoothstep(-0.5, 0.5, vViewPosition.y);
                gl_FragColor = vec4(glowColor, intensity * mask);
            }
        `}
      />
    </mesh>
  );
}

function DynamicSky({ sunPosRef, sunColorRef }) {
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  useFrame(() => {
    if (materialRef.current && sunPosRef.current && sunColorRef.current) {
      materialRef.current.uniforms.uSunPos.value.copy(sunPosRef.current);
      materialRef.current.uniforms.uSunColor.value.copy(sunColorRef.current);
    }
  });

  return (
    <mesh scale={[100, 100, 100]}>
      <sphereGeometry args={[1, 32, 32]} />
      <shaderMaterial 
        ref={materialRef}
        vertexShader={skyVertexShader}
        fragmentShader={skyFragmentShader}
        uniforms={useMemo(() => ({
          uSunPos: { value: new THREE.Vector3() },
          uSunColor: { value: new THREE.Color() }
        }), [])}
        side={THREE.BackSide}
        depthWrite={false}
      />
    </mesh>
  );
}

function TopographicMesh({ sunPos, sunColor, viewMode }) {
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);

  const [colorMap, elevMap] = useTexture([
    'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
    'https://unpkg.com/three-globe/example/img/earth-topology.png'
  ]);

  useFrame((state) => {
    if (materialRef.current?.userData?.shader) {
      if (sunPos) materialRef.current.userData.shader.uniforms.uSunPos.value.copy(sunPos);
      if (sunColor) materialRef.current.userData.shader.uniforms.uSunColor.value.copy(sunColor);
      materialRef.current.userData.shader.uniforms.uViewMode.value = viewMode === 'ar' ? 1.0 : 0.0;
      materialRef.current.userData.shader.uniforms.uTime.value = state.clock.elapsedTime;
    }
  });

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow castShadow>
      {/* High segments for vertex displacement */}
      <planeGeometry args={[10, 10, 512, 512]} />
      <meshStandardMaterial
        ref={materialRef}
        roughness={0.9}
        metalness={0.1}
        bumpMap={elevMap}
        bumpScale={0.06}        onBeforeCompile={(shader) => {
          shader.uniforms.uColorTex = { value: colorMap };
          shader.uniforms.uElevTex = { value: elevMap };
          shader.uniforms.uSunPos = { value: new THREE.Vector3() };
          shader.uniforms.uSunColor = { value: new THREE.Color() };
          shader.uniforms.uViewMode = { value: viewMode === 'ar' ? 1.0 : 0.0 };
          shader.uniforms.uTime = { value: 0 };

          shader.vertexShader = `
            uniform sampler2D uElevTex;
            uniform float uViewMode;
            varying vec2 vTexCoord;
            varying vec3 vWorldPosProc;
            varying vec2 vUvProc;
            
            float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
            float noise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                vec2 u = f*f*(3.0-2.0*f);
                return mix(mix(hash(i + vec2(0.0,0.0)), hash(i + vec2(1.0,0.0)), u.x),
                           mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), u.x), u.y);
            }
            float fbm(vec2 p) {
                float f = 0.0;
                p = p * 4.0;
                f += 0.5000 * noise(p); p = p * 2.1;
                f += 0.2500 * noise(p); p = p * 2.2;
                f += 0.1250 * noise(p); p = p * 2.3;
                f += 0.0625 * noise(p);
                return f;
            }
          ` + shader.vertexShader.replace(
             '#include <begin_vertex>',
             `
             vUvProc = uv;
             vec2 centered = uv - 0.5;
             float dist = length(centered) * 2.0;

             float angle = atan(centered.y, centered.x);
             float lon = angle - 3.14159265 / 2.0; 
             float texU = fract(lon / (2.0 * 3.14159265) + 0.5);
             float texV = 1.0 - dist;
             
             vTexCoord = vec2(texU, texV);

             float elevation = 0.0;
             if (dist <= 1.0) {
               float rawElev = texture2D(uElevTex, vec2(texU, texV)).r;
               
               // Add high-frequency topographic noise for extra realism and tactile feel
               float detailNoise = fbm(uv * 60.0) * mix(0.01, 0.03, uViewMode);
               float fineNoise = fbm(uv * 200.0) * mix(0.002, 0.008, uViewMode);
               
               // Only apply detail to landmass, leaving ocean relatively flat
               float landmask = smoothstep(0.01, 0.06, rawElev);
               float terrainDetail = (detailNoise + fineNoise) * landmask;

               // Non-linear scaling to make peaks sharper and lowlands flatter
               float peakyElev = pow(rawElev, mix(1.0, 1.3, uViewMode));

               // Exaggerate elevation in AR mode for clarity
               float scale = mix(0.3, 0.6, uViewMode);
               
               elevation = peakyElev * scale + terrainDetail;
             }
             if (dist >= 0.9 && dist <= 1.0) {
                float wall = smoothstep(0.9, 0.95, dist);
                // Reduce noise and spikes in AR mode for cleaner analytical rendering
                float spikes = fbm(uv * 15.0) * mix(0.05, 0.015, uViewMode);
                elevation += wall * mix(0.15, 0.08, uViewMode) + spikes * wall;
             }
             if (dist > 1.0) {
                elevation = -1.0; // Drop off
             }
             
             vec3 transformed = vec3(position.xy, position.z + elevation);
             vWorldPosProc = (modelMatrix * vec4(transformed, 1.0)).xyz;
             `
          );

          shader.fragmentShader = `
            uniform sampler2D uColorTex;
            uniform sampler2D uElevTex;
            uniform vec3 uSunPos;
            uniform vec3 uSunColor;
            uniform float uViewMode;
            uniform float uTime;
            varying vec2 vTexCoord;
            varying vec3 vWorldPosProc;
            varying vec2 vUvProc;
          ` + shader.fragmentShader.replace(
             '#include <map_fragment>',
             `
               vec2 centered = vUvProc - 0.5;
               float dist = length(centered) * 2.0;
               if (dist > 1.0) discard;

               vec4 texColor = texture2D(uColorTex, vTexCoord);
               vec3 color = texColor.rgb;
               float el = texture2D(uElevTex, vTexCoord).r;
               float isLand = smoothstep(0.01, 0.05, el);

               // DRONE FOOTAGE COLOR CORRECTION
               if (uViewMode < 0.5) {
                   float contrast = 1.35;
                   color = ((color - 0.5) * max(contrast, 0.0)) + 0.5;
                   
                   vec3 shallowOcean = vec3(0.01, 0.15, 0.45);
                   vec3 deepOcean = vec3(0.005, 0.02, 0.1);
                   float depth = (0.01 - clamp(el, 0.0, 0.01)) * 100.0;
                   vec3 richOcean = mix(deepOcean, shallowOcean, 1.0 - depth) + color * 0.15;
                   color = mix(richOcean, color, isLand);
                   
                   float greenness = color.g - (color.r + color.b) * 0.45;
                   if (greenness > 0.0) {
                      color.g += greenness * isLand * 0.6; 
                      color.r *= 0.85;
                   }
                   
                   float aridness = color.r - (color.g + color.b) * 0.45;
                   if (aridness > 0.0) {
                      color.r += aridness * isLand * 0.4;
                      color.g += aridness * isLand * 0.15;
                   }
               }

               if (dist >= 0.92) {
                  float iceBlend = smoothstep(0.92, 0.97, dist);
                  color = mix(color, vec3(0.8, 0.9, 1.0), iceBlend);
               }
               
               if (uViewMode > 0.5) {
                   // Analytical Topographic Rendering
                   float contour = fract(el * 30.0);
                   float contourLine = smoothstep(0.05, 0.0, contour) + smoothstep(0.95, 1.0, contour);
                   
                   // Distinct landmass vs ocean
                   if (el > 0.05) {
                       color = mix(vec3(0.1, 0.15, 0.25), vec3(0.3, 0.4, 0.35), el); 
                       color += vec3(0.4, 0.7, 0.5) * contourLine * 0.4; 
                   } else {
                       color = vec3(0.02, 0.08, 0.15); // Deep analytical ocean
                   }
                   
                   // Polar coordinate toroidal magnetic field overlay (wrapping from north center around the edge)
                   // The center corresponds to the North Pole.
                   // The dist goes from 0 (North Pole) to 1 (Ice Wall/Edge)
                   // A toroidal field line would go from center (0), flow outwards to 1,
                   // and we animate this flow to create a "centripetal vortex".
                   
                   // Angle around the center (longitude)
                   float angle = atan(centered.y, centered.x);
                   
                   // Field lines spiraling outwards from the North Pole (center) to the edge
                   float spiralBend = 2.0; // How tightly they spiral
                   float numLines = 24.0;
                   
                   // Base phase logic: a wave that flows outwards (dist) and also spirals (angle)
                   // We add uTime to make it flow from center to edge.
                   float fieldPhase = (angle / (2.0 * 3.14159265)) * numLines - dist * spiralBend - uTime * 2.0;
                   float fieldWave = fract(fieldPhase);
                   
                   // Create sharp lines
                   float magneticLine = smoothstep(0.1, 0.0, fieldWave) + smoothstep(0.9, 1.0, fieldWave);
                   
                   // Modulate the intensity of the lines so they pulse/glow
                   float distanceFade = smoothstep(1.0, 0.5, dist) * smoothstep(0.0, 0.2, dist); // Fade at very center and very edge
                   
                   // Toroidal glow:
                   // The field is strong at the center (North pole emission), spreads out, and sinks at the rim.
                   vec3 fieldColor = vec3(0.2, 0.5, 1.0); // Electric blue magnetic field
                   
                   // Add the field overlay to the color
                   color += fieldColor * magneticLine * distanceFade * 0.5;
                   
                   // Add a subtle grid on top of it just for analytical scaling
                   float gridLon = fract((angle / (3.14159265*2.0)) * 12.0);
                   float gridLat = fract(dist * 6.0);
                   float gridLine = smoothstep(0.02, 0.0, gridLon) + smoothstep(0.01, 0.0, gridLat);
                   color += vec3(1.0, 1.0, 1.0) * gridLine * 0.1;
               }
               
               // Dielectric Tide Logic (Electromagnetic aether resonance)
               if (el < 0.05 && dist < 0.95) {
                   float tideDist = length(vWorldPosProc.xz - uSunPos.xz);
                   
                   // Primary standing wave
                   float tideWave = sin(tideDist * -4.0 + uTime * 2.0);
                   float tidePulse = smoothstep(0.7, 1.0, tideWave * 0.5 + 0.5);
                   
                   // Interference harmonics
                   float harmonic = sin(tideDist * -12.0 + uTime * 4.0) * 0.5 + 0.5;
                   float interference = tidePulse * (0.6 + 0.4 * harmonic);
                   
                   // Decay over distance
                   float intensity = interference * smoothstep(8.0, 1.0, tideDist);
                   
                   vec3 tideColor = mix(vec3(0.1, 0.5, 0.9), vec3(0.4, 0.9, 0.6), harmonic);
                   
                   if (uViewMode > 0.5) {
                       // Analytical tide view
                       color = mix(color, vec3(0.2, 0.8, 1.0), intensity * 0.8);
                   } else {
                       // Organic luminescence 
                       color = mix(color, color + tideColor * 2.0, intensity);
                   }
               }
               
               diffuseColor = vec4(color, 1.0);
             `
          ).replace(
             '#include <roughnessmap_fragment>',
             `
               float roughnessFactor = roughness;
               float elMapR = texture2D(uElevTex, vTexCoord).r;
               float isLandR = smoothstep(0.01, 0.05, elMapR);
               if (uViewMode < 0.5) {
                   roughnessFactor = mix(0.15, 0.9, isLandR);
               } else if(el < 0.05 && dist < 0.95) {
                   roughnessFactor = mix(0.3, 0.1, uViewMode);
               } else {
                   roughnessFactor = mix(0.95, 0.8, uViewMode); 
               }
             `
          ).replace(
             '#include <metalnessmap_fragment>',
             `
               float metalnessFactor = metalness;
               float elMapM = texture2D(uElevTex, vTexCoord).r;
               float isLandM = smoothstep(0.01, 0.05, elMapM);
               if (uViewMode < 0.5) {
                  metalnessFactor = mix(0.65, 0.0, isLandM);
               } else if (el < 0.05 && dist < 0.95) {
                  metalnessFactor = mix(0.3, 0.6, uViewMode);
               } else {
                  metalnessFactor = mix(0.05, 0.1, uViewMode); 
               }
             `
          ).replace(
             '#include <normal_fragment_maps>',
             `
             #include <normal_fragment_maps>
             float elMapN = texture2D(uElevTex, vTexCoord).r;
             float isLandN = smoothstep(0.01, 0.05, elMapN);
             
             if (uViewMode < 0.5) {
                 vec3 flatNormal = normalize( vNormal );
                 normal = mix(flatNormal, normal, isLandN);
             } else {
                 normal = normalize( vNormal );
             }
             `
          ).replace(
             '#include <dithering_fragment>',
             `#include <dithering_fragment>
               // Kinetic Shader Logic for Light Morphology: Yin-Yang & Broad-Arc Crescent
               float d_sun = length(vWorldPosProc.xz - uSunPos.xz);
               
               float a_p = atan(vWorldPosProc.z, vWorldPosProc.x);
               float a_s = atan(uSunPos.z, uSunPos.x);
               float r_p = length(vWorldPosProc.xz);
               float r_s = length(uSunPos.xz);
               float d_a = a_p - a_s;
               
               // NODE_65: Light Morphology Specification
               // r_norm = 0 at Inner Circuit (Yin-Yang), 1 at Outer Circuit (Broad-Arc)
               float r_norm = clamp((r_s - 1.5) / 2.0, 0.0, 1.0);
               
               // The spiral bend is high near the center (creating the S-curve), but vanishes at the rim
               float spiralBend = mix(0.4, 0.0, smoothstep(0.4, 0.8, r_norm));
               
               // Darken the center progressively as Sun moves outward to simulate horizontal stretch / rim-light
               float darkCenterFactor = mix(0.0, clamp(4.0 - r_p, 0.0, 4.0) * 0.4, smoothstep(0.5, 0.9, r_norm));
               
               // By adding the bend, we warp the daylight boundary. The center darkness pushes it outwards into an arc.
               float yy_curve = cos(d_a + spiralBend * (r_p - 1.5)) - darkCenterFactor;
               float dayBoundary = smoothstep(-0.05, 0.05, yy_curve);

               // Atmospheric Scattering (Rayleigh & Mie)
               float depth = length(vWorldPosProc - cameraPosition);
               vec3 viewDirFog = normalize(vWorldPosProc - cameraPosition);
               vec3 sunDirFog = normalize(uSunPos - cameraPosition);
               
               float cosThetaFog = dot(viewDirFog, sunDirFog);
               
               float rayleighPhaseFog = 0.75 * (1.0 + cosThetaFog * cosThetaFog);
               float gFog = 0.8;
               float g2Fog = gFog * gFog;
               float miePhaseFog = 1.5 * ((1.0 - g2Fog) / (2.0 + g2Fog)) * (1.0 + cosThetaFog * cosThetaFog) / pow(1.0 + g2Fog - 2.0 * gFog * cosThetaFog, 1.5);
               
               float opticalDepth = 1.0 - exp(-depth * 0.05); 
               vec3 scatterColor = vec3(0.05, 0.15, 0.5) * rayleighPhaseFog + uSunColor * miePhaseFog * 0.01;
               
               vec3 transmission = exp(-vec3(0.05, 0.1, 0.2) * depth * 0.08); 
               
               // Apply Volumetric Diffusion & Light Morphology 
               // Bloom is wider (horizontal stretch) when outer circuit, tighter when inner
               float bloomRadius = mix(12.0, 20.0, r_norm);
               float bloomIntensity = mix(0.15, 0.25, r_norm);
               float bloomLight = smoothstep(bloomRadius, 4.0, d_sun) * bloomIntensity;
               vec3 diffusionGlow = uSunColor * bloomLight;
               
               // Brighten the day side brightness significantly
               vec3 dayColor = mix(gl_FragColor.rgb, gl_FragColor.rgb * 2.0 + diffusionGlow, 0.8);
               
               // Darken the night side so there is high contrast
               vec3 nightColor = gl_FragColor.rgb * vec3(0.1, 0.15, 0.4) + vec3(0.0, 0.02, 0.05);
               
               // Forensically Clear High-Fidelity AR Mode
               if (uViewMode > 0.5) {
                   // Clean, less saturated light for analytical clarity. Preserve topographic details.
                   dayColor = mix(gl_FragColor.rgb, gl_FragColor.rgb * 1.2 + diffusionGlow * 0.1, 0.8);
                   
                   // Analytical ambient base - ensure textures are totally visible in the dark without pure black shadows
                   // Keeping it dark enough for contrast
                   nightColor = gl_FragColor.rgb * vec3(0.3, 0.35, 0.45) + vec3(0.05, 0.08, 0.12);
                   
                   // Greatly reduce atmospheric transmission loss so objects don't vanish in fog
                   transmission = mix(transmission, vec3(1.0), 0.7);
                   opticalDepth *= 0.1; // Vastly reduce aggressive scattering
                   
                   // Add subtle rim lighting at the edges for volumetric definition
                   float edgeGlow = smoothstep(0.6, 0.0, abs(dot(viewDirFog, vec3(0.0, 1.0, 0.0))));
                   nightColor += vec3(0.05, 0.1, 0.15) * edgeGlow * (1.0 - dayBoundary);
                   dayColor += vec3(0.05, 0.1, 0.15) * edgeGlow;
               }
               
               vec3 baseColor = mix(nightColor, dayColor, dayBoundary);

               gl_FragColor.rgb = baseColor * transmission + scatterColor * opticalDepth;
             `
          );
          
          if (materialRef.current) {
             materialRef.current.userData.shader = shader;
          }
        }}
      />
    </mesh>
  );
}

const farmNodeVertexShader = `
  varying vec2 vUv;
  varying vec3 vWorldPosition;
  void main() {
    vUv = uv;
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const farmNodeFragmentShader = `
  varying vec2 vUv;
  varying vec3 vWorldPosition;
  uniform vec3 uSunPos;
  uniform vec3 uNodePos;
  uniform float uTime;
  
  void main() {
    // Distance from this hex node to the Sun's projection
    float distToSun = distance(vWorldPosition, vec3(uSunPos.x, 0.0, uSunPos.z));
    
    // Base Amber (Standby)
    vec3 standbyColor = vec3(0.8, 0.4, 0.0);
    // Active Green (Peak Exposure)
    vec3 activeColor = vec3(0.1, 0.9, 0.3);
    
    // Activation ranges (starts responding at 2.5 distance, peaks at 0.5)
    float activation = smoothstep(2.5, 0.5, distToSun);
    
    vec3 finalColor = mix(standbyColor, activeColor, activation);
    
    // Pulse calculation based on activation state
    float pulseSpeed = mix(1.0, 5.0, activation);
    float pulse = sin(uTime * pulseSpeed + (uNodePos.x * 10.0)) * 0.5 + 0.5;
    float glow = mix(0.4, 1.0, activation * pulse);
    
    // Hexagon UV styling
    vec2 centered = vUv - 0.5;
    float d = length(centered);
    
    if (d > 0.4) {
       // Outer Hex Outline
       finalColor *= 1.5;
       glow *= 1.2;
    } else if (d < 0.1) {
       // Core indicator
       finalColor = mix(vec3(1.0), activeColor, 0.5);
       glow = 1.0;
    } else {
       // Interior body
       glow *= 0.3;
    }
    
    float alpha = glow * mix(0.5, 1.0, activation);
    if (alpha <= 0.05) discard;
    
    gl_FragColor = vec4(finalColor * glow, alpha);
  }
`;

function FarmNodes({ sunPosRef }) {
  const nodes = useMemo(() => {
    const list = [];
    // Generating 42 representation nodes in the farming bands
    for (let i = 0; i < 42; i++) {
       const angle = Math.random() * Math.PI * 2;
       // Distribute between Tropic of Cancer (1.25) and Capricorn (3.75)
       const radius = 1.25 + Math.random() * 2.5; 
       list.push(new THREE.Vector3(radius * Math.cos(angle), 0.03, radius * Math.sin(angle)));
    }
    return list;
  }, []);
  
  return (
    <group>
      {nodes.map((pos, i) => (
         <FarmNode key={i} position={pos} sunPosRef={sunPosRef} />
      ))}
    </group>
  );
}

function FarmNode({ position, sunPosRef }) {
  const matRef = useRef<THREE.ShaderMaterial>(null);
  
  useFrame((state) => {
    if (matRef.current && sunPosRef.current) {
       matRef.current.uniforms.uSunPos.value.copy(sunPosRef.current);
       matRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
  });
  
  return (
     <mesh position={position} rotation={[-Math.PI / 2, 0, Math.PI / 6]} castShadow={false} receiveShadow={false}>
        <circleGeometry args={[0.05, 6]} /> {/* 6 segments makes a hexagon */}
        <shaderMaterial 
           ref={matRef}
           vertexShader={farmNodeVertexShader}
           fragmentShader={farmNodeFragmentShader}
           transparent
           depthWrite={false}
           blending={THREE.AdditiveBlending}
           uniforms={useMemo(() => ({
              uSunPos: { value: new THREE.Vector3() },
              uNodePos: { value: position },
              uTime: { value: 0 },
           }), [position])}
        />
     </mesh>
  );
}
function SunMoon({ time, metricsRef, sunPosRef, sunColorRef, viewMode }) {
  const sunRef = useRef<THREE.Mesh>(null);
  const moonRef = useRef<THREE.Mesh>(null);
  const lightRef = useRef<THREE.PointLight>(null);
  const bloomRef = useRef<THREE.Group>(null);
  const beamRef = useRef<THREE.Mesh>(null);
  const mapMarkerRef = useRef<THREE.Mesh>(null);
  const bloomMatRef = useRef<THREE.ShaderMaterial>(null);
  const phaseVisualRef = useRef<HTMLDivElement>(null);
  const phaseTextRef = useRef<HTMLDivElement>(null);
  const moonContainerRef = useRef<HTMLDivElement>(null);
  
  const sunMatRef = useRef<THREE.ShaderMaterial>(null);
  const moonMatRef = useRef<THREE.MeshStandardMaterial>(null);
  
  const { viewport } = useThree();

  useFrame((state) => {
    const rEquator = 2.5;
    const rCancer = 1.25;
    const rCapricorn = 3.75;
    
    const FULL_CYCLE_DAYS = 365.25;
    const normalDays = 365;
    const yearDay = time.current.time % FULL_CYCLE_DAYS;
    const isReset = false;
    
    // Epoch Event Aurora Discharge
    if (isReset) {
      sunColorRef.current.set('#60a5fa');
      if (bloomMatRef.current) {
        bloomMatRef.current.uniforms.uColor.value.set('#60a5fa');
        bloomMatRef.current.uniforms.uOpacity.value = 1.0 + Math.sin(state.clock.elapsedTime * 10) * 0.5;
      }
      if (bloomRef.current) {
        bloomRef.current.scale.setScalar(2.5 + Math.sin(state.clock.elapsedTime * 20) * 0.5);
      }
    } else {
      sunColorRef.current.set('#fef08a');
      if (bloomMatRef.current) {
        bloomMatRef.current.uniforms.uColor.value.set('#fef08a');
        bloomMatRef.current.uniforms.uOpacity.value = 0.5;
      }
      if (bloomRef.current) {
        bloomRef.current.scale.setScalar(1.0);
      }
    }
    
    let sunRadius = rEquator;
    const sunAngle = Math.PI / 2 + (time.current.time % 1.0) * Math.PI * 2;
    let lagRad = 0;
    let moonAngle = 0;

    if (isReset) {
      sunRadius = rEquator;
      moonAngle = sunAngle;
    } else {
      sunRadius = rEquator + (rCapricorn - rEquator) * Math.cos((yearDay - 355) / 365.25 * Math.PI * 2);
      lagRad = yearDay * (Math.PI * 2 / 28);
      moonAngle = sunAngle - lagRad;
    }

    const sx = Math.cos(sunAngle) * sunRadius;
    const sy = Math.sin(sunAngle) * sunRadius;
    // Keep altitude constant in 3D
    const altitude = 1.0;

    let mx = sx, my = sy;
    if (!isReset) {
      mx = Math.cos(moonAngle) * sunRadius;
      my = Math.sin(moonAngle) * sunRadius;
    }

    if (sunRef.current) sunRef.current.position.set(sx, altitude, sy);
    if (lightRef.current) {
        lightRef.current.position.set(sx, altitude, sy);
        // Shadow Melt Effect: sharper closer to center, softer and wider at rim
        const r_s = Math.sqrt(sx*sx + sy*sy);
        const rNorm = Math.max(0, Math.min(1, (r_s - 1.5) / 2.0));
        lightRef.current.shadow.radius = 1.0 + rNorm * 10.0;
    }

    if (moonRef.current) {
      if (isReset) {
        moonRef.current.position.set(sx, altitude-0.1, sy);
      } else {
        moonRef.current.position.set(mx, altitude-0.2, my);
      }
    }
    
    if (beamRef.current) {
        const sunPos = new THREE.Vector3(sx, altitude, sy);
        const moonPos = isReset ? new THREE.Vector3(sx, altitude-0.1, sy) : new THREE.Vector3(mx, altitude-0.2, my);
        const dist = sunPos.distanceTo(moonPos);
        beamRef.current.scale.y = dist;
        beamRef.current.position.copy(sunPos).lerp(moonPos, 0.5);
        if (dist > 0.001) {
            beamRef.current.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), moonPos.clone().sub(sunPos).normalize());
        }
    }
    
    if (mapMarkerRef.current) {
        mapMarkerRef.current.position.set(mx, 0.02, my);
        const pulseScale = 1.0 + Math.sin(state.clock.elapsedTime * 3) * 0.15;
        mapMarkerRef.current.scale.setScalar(pulseScale);
    }

    // Dynamic sun color & size
    const zoomRatio = (rCapricorn - sunRadius) / (rCapricorn - rCancer);
    const zoomEffect = Math.pow(zoomRatio, 1.8);
    
    const r = 1.0;
    const g = (120 + zoomEffect * 135) / 255.0;
    const b = (50 + zoomEffect * 205) / 255.0;
    
    const sunColor = isReset ? new THREE.Color(0.06, 0.72, 0.5) : new THREE.Color(r, g, b);
    if (lightRef.current) lightRef.current.color.copy(sunColor);
    
    if (sunPosRef && sunPosRef.current) {
        sunPosRef.current.set(sx, altitude, sy);
    }
    if (sunColorRef && sunColorRef.current) {
        sunColorRef.current.copy(sunColor);
    }

    if (sunMatRef.current) {
        sunMatRef.current.uniforms.uColor.value.copy(sunColor);
        sunMatRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
    if (moonMatRef.current?.userData?.shader) {
        moonMatRef.current.userData.shader.uniforms.uTime.value = state.clock.elapsedTime;
        if (sunRef.current) {
            moonMatRef.current.userData.shader.uniforms.uSunPos.value.copy(sunRef.current.position);
        }
    }
    if (sunRef.current) sunRef.current.scale.setScalar(0.1 + zoomEffect * 0.1);

    if (bloomRef.current && bloomMatRef.current) {
      bloomRef.current.position.set(sx, altitude, sy);
      bloomRef.current.visible = true;
      const pulse = Math.sin(performance.now() / 150) * 0.05;
      const effectScale = 1.0 + zoomEffect * 1.5 + pulse;
      bloomRef.current.scale.setScalar(effectScale);
      bloomMatRef.current.uniforms.uColor.value.copy(sunColor);
      bloomMatRef.current.uniforms.uOpacity.value = isReset ? 0.3 : 0.5 + 0.3 * zoomEffect;
    }

    // Calculate phase mathematically for both HTML visualizer and the UI metrics
    let phase = 'New Moon';
    const lagMod = (lagRad % (Math.PI * 2)) / (Math.PI * 2);
    if (lagMod > 0.05 && lagMod < 0.45) phase = 'Waxing Crescent';
    else if (lagMod > 0.45 && lagMod < 0.55) phase = 'Full Moon';
    else if (lagMod > 0.55 && lagMod < 0.95) phase = 'Waning Crescent';

    // Update HTML overlay
    if (phaseTextRef.current) {
       phaseTextRef.current.innerText = isReset ? 'RESET' : phase;
    }
    if (phaseVisualRef.current) {
       const MOON_SIZE = 24;
       const CLEAR_OFFSET = MOON_SIZE * 1.2;
       let shadowTranslate = 0;
       if (lagMod <= 0.5) {
          shadowTranslate = -(lagMod / 0.5) * CLEAR_OFFSET;
       } else {
          shadowTranslate = CLEAR_OFFSET - ((lagMod - 0.5) / 0.5) * CLEAR_OFFSET;
       }
       if (isReset) shadowTranslate = 0; // Reset looks like a new moon visual
       phaseVisualRef.current.style.transform = `translateX(${shadowTranslate}px) scale(1.15)`;
       
       if (moonContainerRef.current) {
          const illumination = Math.abs(shadowTranslate) / CLEAR_OFFSET;
          moonContainerRef.current.style.boxShadow = `0 0 ${8 + 12 * illumination}px rgba(255, 255, 255, ${0.2 + 0.6 * illumination})`;
       }
    }

    // Update metrics ref for UI
    if (metricsRef.current) {
      let stateStr = 'EQNX-CALIB';
      if (isReset) stateStr = 'DIELECTRIC DISCHARGE';
      else if (zoomRatio > 0.7) stateStr = 'SUMMER-ZOOM';
      else if (zoomRatio < 0.3) stateStr = 'WINTER-FOG';

      metricsRef.current({
        day: Math.floor(yearDay),
        solarRadius: Math.floor((sunRadius / rCapricorn) * 12400),
        lunarLag: Math.floor((lagRad % (Math.PI * 2)) * 180 / Math.PI),
        phase: isReset ? 'RESET' : phase,
        state: stateStr,
        isReset: isReset,
        sunPos: new THREE.Vector3(sx, altitude, sy),
        sunColor: sunColor,
        zoomRatio: zoomRatio
      });
    }
  });

  const bloomUniforms = useMemo(() => ({
    uColor: { value: new THREE.Color() },
    uOpacity: { value: 0.0 }
  }), []);

  return (
    <>
      {/* Sun Light */}
      <directionalLight
         ref={lightRef}
         intensity={2.5}
         castShadow
         shadow-bias={-0.0005}
         shadow-normalBias={0.02}
         shadow-mapSize={[8192, 8192]}
         shadow-camera-near={0.5}
         shadow-camera-far={30}
         shadow-camera-left={-10}
         shadow-camera-right={10}
         shadow-camera-top={10}
         shadow-camera-bottom={-10}
         shadow-radius={2}
      />

      {/* Bloom Mesh (Volumetric Halo) */}
      <Billboard ref={bloomRef}>
         <mesh>
            <planeGeometry args={[5, 5]} />
            <shaderMaterial 
               ref={bloomMatRef}
               vertexShader={bloomVertexShader}
               fragmentShader={bloomFragmentShader}
               uniforms={bloomUniforms}
               transparent
               depthWrite={false}
               blending={THREE.AdditiveBlending}
            />
         </mesh>
      </Billboard>
      
      {/* Sun Mesh */}
      <mesh ref={sunRef}>
        <sphereGeometry args={[1, 32, 32]} />
        <shaderMaterial 
           ref={sunMatRef}
           vertexShader={sunVertexShader}
           fragmentShader={sunFragmentShader}
           uniforms={useMemo(() => ({
              uColor: { value: new THREE.Color() },
              uTime: { value: 0.0 }
           }), [])}
        />
      </mesh>
      
      {/* Moon Mesh */}
      <mesh ref={moonRef}>
        <sphereGeometry args={[0.08, 32, 32]} />
        <meshStandardMaterial 
           ref={moonMatRef}
           color="#ffffff" 
           emissive="#ffffff" 
           emissiveIntensity={1.5} 
           roughness={0.2} 
           metalness={0.9} 
           onBeforeCompile={(shader) => {
              shader.uniforms.uTime = { value: 0 };
              shader.uniforms.uSunPos = { value: new THREE.Vector3() };
              shader.vertexShader = `
                varying vec2 vUvProc;
                varying vec3 vWorldPositionProc;
              ` + shader.vertexShader.replace(
                '#include <uv_vertex>',
                `#include <uv_vertex>
                 vUvProc = uv;
                 vWorldPositionProc = (modelMatrix * vec4(position, 1.0)).xyz;`
              );
              shader.fragmentShader = `
                uniform float uTime;
                uniform vec3 uSunPos;
                varying vec2 vUvProc;
                varying vec3 vWorldPositionProc;

                // 3D Simplex noise functions
                vec4 permuteProc(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
                float snoiseProc(vec3 v){
                  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
                  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
                  vec3 i  = floor(v + dot(v, C.yyy));
                  vec3 x0 = v - i + dot(i, C.xxx);
                  vec3 g = step(x0.yzx, x0.xyz);
                  vec3 l = 1.0 - g;
                  vec3 i1 = min(g.xyz, l.zxy);
                  vec3 i2 = max(g.xyz, l.zxy);
                  vec3 x1 = x0 - i1 + C.xxx;
                  vec3 x2 = x0 - i2 + C.yyy;
                  vec3 x3 = x0 - D.yyy;
                  i = mod(i, 289.0);
                  vec4 p = permuteProc(permuteProc(permuteProc(
                             i.z + vec4(0.0, i1.z, i2.z, 1.0))
                           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
                  float n_ = 0.142857142857;
                  vec3 ns = n_ * D.wyz - D.xzx;
                  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
                  vec4 x_ = floor(j * ns.z);
                  vec4 y_ = floor(j - 7.0 * x_ );
                  vec4 x = x_ * ns.x + ns.yyyy;
                  vec4 y = y_ * ns.x + ns.yyyy;
                  vec4 h = 1.0 - abs(x) - abs(y);
                  vec4 b0 = vec4(x.xy, y.xy);
                  vec4 b1 = vec4(x.zw, y.zw);
                  vec4 s0 = floor(b0) * 2.0 + 1.0;
                  vec4 s1 = floor(b1) * 2.0 + 1.0;
                  vec4 sh = -step(h, vec4(0.0));
                  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
                  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
                  vec3 p0 = vec3(a0.xy, h.x);
                  vec3 p1 = vec3(a0.zw, h.y);
                  vec3 p2 = vec3(a1.xy, h.z);
                  vec3 p3 = vec3(a1.zw, h.w);
                  vec4 norm = inversesqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
                  p0 *= norm.x;
                  p1 *= norm.y;
                  p2 *= norm.z;
                  p3 *= norm.w;
                  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
                  m = m * m;
                  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
                }
              ` + shader.fragmentShader.replace(
                '#include <map_fragment>',
                `#include <map_fragment>
                 float n1 = snoiseProc(vec3(vUvProc * 10.0, uTime * 0.05));
                 float n2 = snoiseProc(vec3(vUvProc * 20.0, uTime * 0.1));
                 float noiseValue = (n1 + n2 * 0.5) * 0.5 + 0.5;
                 
                 // Apply subtle crater-like procedural patterns to the base color
                 diffuseColor.rgb *= mix(0.5, 1.2, noiseValue);
                `
              ).replace(
                '#include <dithering_fragment>',
                `#include <dithering_fragment>
                 
                 vec3 viewDir = normalize(cameraPosition - vWorldPositionProc);
                 vec3 lightDir = normalize(uSunPos - vWorldPositionProc);
                 
                 // Perspective Scalar Calibration (Angle of Exposure)
                 float incidenceAngle = max(0.0, viewDir.y);
                 float horizonFactor = 1.0 - incidenceAngle;
                 
                 // The "face" of the Moon observed from the ground is a Refractive Projection
                 // based on intersection of Sun's light and observer's line of sight.
                 float dielectricGlow = max(0.0, dot(vNormal, lightDir));
                 float observerExcitation = max(0.0, dot(vNormal, viewDir));
                 
                 // Compress the phase to simulate the "flattened" perspective of the firmament
                 // Adjust phase mask using horizonFactor. Closer to horizon -> more widened/flattened dielectric reflection
                 float phaseMask = pow(dielectricGlow, mix(1.5, 0.5, horizonFactor)) * observerExcitation;
                 
                 // Deep blue underlying dielectric structure
                 vec3 baseDielectric = vec3(0.02, 0.05, 0.2);
                 // bright plasma face
                 vec3 plasmaColor = vec3(0.95, 0.98, 1.0);
                 
                 // Mix the final color to overwrite the standard lighting
                 // Heightened intensity near the horizon (Atmospheric Zoom)
                 float intensityBoost = 1.0 + horizonFactor * 1.5;
                 vec3 finalMoonColor = mix(baseDielectric, plasmaColor, phaseMask * 2.0 * intensityBoost + 0.1);
                 
                 // Add atmospheric red-shifting near the horizon
                 float redshift = smoothstep(0.4, 1.0, horizonFactor);
                 finalMoonColor.r += redshift * 0.3 * phaseMask;
                 finalMoonColor.b -= redshift * 0.2 * phaseMask;
                 
                 // Add procedural crater details back into the plasma
                 finalMoonColor *= mix(0.6, 1.1, noiseValue);
                 
                 gl_FragColor = vec4(finalMoonColor, 1.0);
                `
              );
              if (moonMatRef.current) {
                 moonMatRef.current.userData.shader = shader;
              }
           }}
        />
        <Html position={[0, 0.25, 0]} center style={{ pointerEvents: 'none', transition: 'all 0.1s' }}>
          <div className="flex flex-col items-center gap-1 opacity-90 transition-opacity">
            <div ref={moonContainerRef} className="w-6 h-6 rounded-full bg-slate-200 relative overflow-hidden shadow-[0_0_10px_rgba(255,255,255,0.4)] border border-white/20">
               {/* Surface texture */}
               <div className="absolute inset-0 opacity-40 mix-blend-multiply">
                  <div className="absolute top-1 left-3 w-1.5 h-1.5 bg-slate-500 rounded-full blur-[0.5px]"></div>
                  <div className="absolute top-3 left-4 w-2 h-2 bg-slate-500 rounded-full blur-[0.5px]"></div>
                  <div className="absolute top-4 left-1 w-1 h-1 bg-slate-500 rounded-full blur-[0.5px]"></div>
               </div>
               {/* The moving shadow representing the phase */}
               <div ref={phaseVisualRef} className="absolute inset-0 bg-slate-900 rounded-full" style={{ transition: 'transform 0.05s linear' }} />
               {/* Inner glow/atmosphere to simulate lighting */}
               <div className="absolute inset-0 rounded-full shadow-[inset_-2px_-2px_4px_rgba(0,0,0,0.4)] border border-white/10" />
            </div>
            <div ref={phaseTextRef} className="bg-black/60 text-[8px] font-mono tracking-widest text-zinc-100 px-1.5 py-0.5 rounded backdrop-blur-md border border-white/10 uppercase" />
          </div>
        </Html>
      </mesh>

      {/* Optical Convergence Beam */}
      <mesh ref={beamRef} castShadow={false} receiveShadow={false}>
        <cylinderGeometry args={[0.01, 0.01, 1, 8]} />
        <meshBasicMaterial color="#60a5fa" transparent opacity={0.3} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>

      {/* Convergence Source on Map */}
      <mesh ref={mapMarkerRef} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.02, 0.05, 16]} />
        <meshBasicMaterial color="#93c5fd" transparent opacity={0.8} blending={THREE.AdditiveBlending} side={THREE.DoubleSide} />
        <Html position={[0, 0, 0]} center style={{ pointerEvents: 'none' }}>
           <div className="w-8 h-8 rounded-full border border-blue-500/50 flex items-center justify-center -translate-y-1/2 -translate-x-1/2">
             <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse shadow-[0_0_8px_4px_rgba(96,165,250,0.4)]" />
           </div>
        </Html>
      </mesh>
    </>
  );
}

const Controls = ({ viewMode }) => {
  const controlsRef = useRef<any>(null);
  const { camera } = useThree();

  useFrame(() => {
    if (controlsRef.current && viewMode !== 'ar') {
      if (viewMode === 'top') {
        // Ortho-Topographic 90 deg
        controlsRef.current.setPolarAngle(0);
        controlsRef.current.setAzimuthalAngle(0);
        controlsRef.current.target.set(0, 0, 0);
      } else {
        // Observer-Perspective 45 deg
        controlsRef.current.minPolarAngle = Math.PI / 4;
        controlsRef.current.maxPolarAngle = Math.PI / 2.5;
        controlsRef.current.target.set(0, 0, 0);
      }
    }
  });

  React.useEffect(() => {
    if (viewMode === 'ar') {
       // Move camera to center observer spot for AR, looking UP
       camera.position.set(0, 0.5, 0);
    } else {
       // Reset camera
       camera.position.set(0, 8, 4);
    }
  }, [viewMode, camera]);

  if (viewMode === 'ar') {
    return <DeviceOrientationControls />;
  }

  return <OrbitControls ref={controlsRef} enablePan={false} enableZoom={true} minDistance={2} maxDistance={15} enableDamping={true} dampingFactor={0.05} />;
};


function DensityAltimeter() {
  const [altitude, setAltitude] = React.useState(0);

  // Mock barometer/altitude for demo
  useFrame((state) => {
    // Oscillate between 0 and 1500 meters slowly
    const alt = 750 + Math.sin(state.clock.elapsedTime * 0.2) * 750;
    setAltitude(alt);
  });

  // Map altitude (0 - 1500m) to index of refraction (n) from 1.00030 to 1.00010 approx
  const n0 = 1.000293;
  const dn_dh = -0.000022; // rough gradient per 1000m 
  const current_n = (n0 + (altitude / 1000) * dn_dh).toFixed(5);
  
  // Normalized for UI bar 0-1
  const fillPerc = Math.max(0, Math.min(100, (altitude / 1500) * 100));

  return (
    <Html 
      center
      style={{
        position: 'absolute',
        top: '-100px',
        left: '170px', // Places it on the right edge of the watch face ~ diameter/2
        pointerEvents: 'none'
      }}
    >
      <div className="flex bg-black/60 border border-white/10 rounded-full h-[200px] w-12 flex-col items-center py-4 justify-between backdrop-blur-md shadow-[0_0_15px_rgba(0,0,0,0.8)]">
        <div className="text-[9px] font-mono text-zinc-400">Idx(n)</div>
        
        <div className="relative h-full w-2 bg-zinc-900 rounded-full my-2 overflow-hidden flex flex-col justify-end">
           {/* Ticks for UI */}
           <div className="absolute inset-0 flex flex-col justify-between py-1 z-10 w-full opacity-30">
               {[...Array(10)].map((_, i) => <div key={i} className="h-px bg-white w-full" />)}
           </div>
           {/* Fill */}
           <div 
             className="w-full bg-gradient-to-t from-blue-600 via-emerald-500 to-amber-500"
             style={{ height: `${fillPerc}%`, transition: 'height 0.1s linear' }}
           />
        </div>

        <div className="flex flex-col items-center gap-0.5">
            <span className="text-[10px] font-mono tracking-tighter text-emerald-400">{current_n}</span>
            <span className="text-[8px] font-mono text-zinc-500">{(altitude).toFixed(0)}m</span>
        </div>
      </div>
    </Html>
  );
}

function NSSPostProcessing({ viewMode }) {
  const { gl, scene, camera, size } = useThree();
  const bloomPassRef = useRef<any>(null);
  const caPassRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  if (!canvasRef.current) {
     canvasRef.current = document.createElement('canvas');
     canvasRef.current.width = 1;
     canvasRef.current.height = 1;
  }

  useFrame(() => {
    if (viewMode !== 'ar') return;
    
    // Sample the center of the video feed to determine ambient light luminance
    const video = document.querySelector('video');
    if (video && video.readyState >= 2 && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
        if (ctx) {
            // Read 1px from the center of the video feed
            ctx.drawImage(video, video.videoWidth / 2, video.videoHeight / 2, 1, 1, 0, 0, 1, 1);
            const data = ctx.getImageData(0, 0, 1, 1).data;
            const luminance = (data[0] * 0.299 + data[1] * 0.587 + data[2] * 0.114) / 255.0;

            // Forensically clear logic:
            // High luminance (bright environment) -> Reduce bloom/CA so graphics remain sharp
            // Low luminance (dark environment) -> Increase bloom and CA for volumetric pop and "night vision" vibe
            const targetBloom = THREE.MathUtils.lerp(0.8, 0.3, luminance);
            const targetCa = THREE.MathUtils.lerp(0.003, 0.0005, luminance);

            if (bloomPassRef.current) {
                bloomPassRef.current.strength = THREE.MathUtils.lerp(bloomPassRef.current.strength, targetBloom, 0.05);
            }
            if (caPassRef.current) {
                caPassRef.current.uniforms.amount.value = THREE.MathUtils.lerp(caPassRef.current.uniforms.amount.value, targetCa, 0.05);
            }
        }
    }
  });

  if (viewMode !== 'ar') return null;

  return (
    <Effects disableGamma>
      <tAARenderPass attachArray="passes" args={[scene, camera, 0x000000, 0]} sampleLevel={2} />
      <unrealBloomPass ref={bloomPassRef} attachArray="passes" args={[new THREE.Vector2(size.width, size.height), 0.6, 0.4, 0.8]} />
      <shaderPass ref={caPassRef} attachArray="passes" args={[ChromaticAberrationShader]} />
    </Effects>
  );
}

const ChromaticAberrationShader = {
  uniforms: {
    tDiffuse: { value: null },
    amount: { value: 0.0008 },
    angle: { value: Math.PI / 4 }
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float amount;
    uniform float angle;
    varying vec2 vUv;
    void main() {
      vec2 offset = amount * vec2( cos(angle), sin(angle));
      vec4 cr = texture2D(tDiffuse, vUv + offset);
      vec4 cga = texture2D(tDiffuse, vUv);
      vec4 cb = texture2D(tDiffuse, vUv - offset);
      gl_FragColor = vec4(cr.r, cga.g, cb.b, cga.a);
    }
  `
};

function MagneticLeyLines({ time, viewMode }) {
  const groupRef = useRef<THREE.Group>(null);
  
  const numLines = 36;
  const numPoints = 128;
  const radius = 5.0; 
  
  const lines = useMemo(() => {
    const arr = [];
    for (let i = 0; i < numLines; i++) {
      const angle = (i / numLines) * Math.PI * 2;
      const pts = [];
      for (let j = 0; j <= numPoints; j++) {
        const t = j / numPoints; 
        const r = t * radius;
        // height arcs up and then down
        const h = Math.sin(t * Math.PI) * 1.5; 
        const spiralOffset = t * Math.PI * 0.5; // Slight spiral
        const x = Math.cos(angle + spiralOffset) * r;
        const z = Math.sin(angle + spiralOffset) * r;
        pts.push(new THREE.Vector3(x, h, z));
      }
      arr.push(new THREE.CatmullRomCurve3(pts));
    }
    return arr;
  }, []);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * -0.05; // Slowly rotating field
      
      // Update shader uniforms
      groupRef.current.children.forEach(child => {
         if (child.material && child.material.uniforms) {
             child.material.uniforms.uTime.value = state.clock.elapsedTime;
         }
      });
    }
  });

  if (viewMode === 'top') return null;

  return (
    <group ref={groupRef} position={[0, 0.05, 0]}>
      {lines.map((curve, idx) => {
        // Individual phase for each line so they don't pulse at the same time
        const phaseOffset = (idx / numLines) * 2.0 * Math.PI;
        return (
        <mesh key={idx}>
          <tubeGeometry args={[curve, 64, 0.008, 8, false]} />
          <shaderMaterial 
             transparent 
             blending={THREE.AdditiveBlending}
             depthWrite={false}
             uniforms={{
                uTime: { value: 0 },
                uPhase: { value: phaseOffset },
                uColor: { value: new THREE.Color("#1e90ff") },
                uPulseColor: { value: new THREE.Color("#b3d9ff") }
             }}
             vertexShader={`
                varying vec2 vUv;
                void main() {
                   vUv = uv;
                   gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
             `}
             fragmentShader={`
                uniform float uTime;
                uniform float uPhase;
                uniform vec3 uColor;
                uniform vec3 uPulseColor;
                varying vec2 vUv;
                void main() {
                   // vUv.x represents the length along the tube
                   float pulse = fract(vUv.x * 3.0 - uTime * 1.5 + uPhase);
                   pulse = smoothstep(0.4, 0.5, pulse) * smoothstep(0.6, 0.5, pulse); // Sharp leading edge, trailing tail
                   
                   float intensity = mix(0.1, 1.0, pulse) * smoothstep(1.0, 0.8, vUv.x) * smoothstep(0.0, 0.2, vUv.x);
                   
                   vec3 finalColor = mix(uColor, uPulseColor, pulse);
                   gl_FragColor = vec4(finalColor, intensity * 0.8);
                }
             `}
          />
        </mesh>
      )})}
      {/* Central emitting node */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.08, 32, 32]} />
        <meshBasicMaterial color="#4da6ff" transparent opacity={0.8} blending={THREE.AdditiveBlending} />
      </mesh>
      {/* Outer gathering rim */}
      <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, 0, 0]}>
        <ringGeometry args={[radius - 0.05, radius, 128]} />
        <meshBasicMaterial color="#1e90ff" transparent opacity={0.4} blending={THREE.AdditiveBlending} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

export default function Sovereign3D({ simState, setMetrics, viewMode, showJourney }) {
  const metricsRef = useRef((data) => {
    setMetrics(data);
  });

  // State to pass from frame loop to topological mesh without re-renders
  const sunPosRef = useRef(new THREE.Vector3());
  const sunColorRef = useRef(new THREE.Color());

  return (
    <Canvas shadows camera={{ position: [0, 8, 0], fov: 45 }} gl={{ antialias: true, alpha: true }}>
      {viewMode !== 'ar' && <color attach="background" args={['#010103']} />}
      {viewMode !== 'ar' && <fog attach="fog" args={['#010206', 4, 25]} />}
      <ambientLight intensity={viewMode === 'ar' ? 0.3 : 0.2} color="#8ab4f8" />
      <hemisphereLight intensity={viewMode === 'ar' ? 0.4 : 0.4} color="#4488ff" groundColor="#111111" />
      
      {viewMode !== 'ar' && <AtmospherePolish />}
      {viewMode !== 'ar' && <DynamicSky sunPosRef={sunPosRef} sunColorRef={sunColorRef} />}
      {viewMode !== 'ar' && <DynamicStarfield sunPosRef={sunPosRef} />}
      
      <Suspense fallback={null}>
        <TopographicMesh sunPos={sunPosRef.current} sunColor={sunColorRef.current} viewMode={viewMode} />
      </Suspense>
      <MagneticLeyLines time={simState} viewMode={viewMode} />
      <FarmNodes sunPosRef={sunPosRef} />
      <SunMoon time={simState} metricsRef={metricsRef} sunPosRef={sunPosRef} sunColorRef={sunColorRef} viewMode={viewMode} />
      {showJourney && <CircuitMesh time={simState} />}
      
      {viewMode === 'ar' && <ARSkyGrid time={simState} sunPosRef={sunPosRef} sunColorRef={sunColorRef} />}
      <DensityAltimeter />
      <Controls viewMode={viewMode} />
      <NSSPostProcessing viewMode={viewMode} />
    </Canvas>
  );
}

function ARSkyGrid({ time, sunPosRef, sunColorRef }) {
  const rCancer = 1.25;
  const rEquator = 2.5;
  const rCapricorn = 3.75;
  const altitude = 1.0;
  
  const horizonMatRef = useRef<THREE.ShaderMaterial>(null);
  const chladniMatRef = useRef<THREE.ShaderMaterial>(null);

  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 512;
        source.connect(analyser);
        analyserRef.current = analyser;
        dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
        audioCtxRef.current = audioCtx;
      } catch (err) {
        console.warn("Acoustic Scanner (Mic) access denied or unavailable:", err);
      }
    };
    initAudio();
    return () => {
       if (audioCtxRef.current) {
         audioCtxRef.current.close().catch(console.error);
       }
    };
  }, []);

  useFrame((state) => {
     if (horizonMatRef.current && sunPosRef?.current && sunColorRef?.current) {
        horizonMatRef.current.uniforms.uSunPos.value.copy(sunPosRef.current);
        horizonMatRef.current.uniforms.uSunColor.value.copy(sunColorRef.current);
     }
     
     let h432 = 0;
     let h528 = 0;
     let chaos = 0;

     if (analyserRef.current && dataArrayRef.current) {
        analyserRef.current.getByteFrequencyData(dataArrayRef.current);
        const data = dataArrayRef.current;
        
        // Approximate bins based on sample rate ~44100Hz / 512 = ~86Hz per bin
        // 432Hz -> bin 5 (430Hz)
        // 528Hz -> bin 6 (516Hz)
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i];
        const avg = sum / data.length;
        
        h432 = data[5] ? data[5] / 255.0 : 0;
        h528 = data[6] ? data[6] / 255.0 : 0;
        chaos = avg / 255.0;

        // Isolate harmonics from chaos
        if (h432 > chaos * 1.5) h432 *= 1.2; else h432 *= 0.1;
        if (h528 > chaos * 1.5) h528 *= 1.2; else h528 *= 0.1;
     }

     if (chladniMatRef.current) {
        chladniMatRef.current.uniforms.uTime.value = state.clock.elapsedTime;
        chladniMatRef.current.uniforms.uHarmonic432.value = THREE.MathUtils.lerp(chladniMatRef.current.uniforms.uHarmonic432.value, h432, 0.1);
        chladniMatRef.current.uniforms.uHarmonic528.value = THREE.MathUtils.lerp(chladniMatRef.current.uniforms.uHarmonic528.value, h528, 0.1);
        chladniMatRef.current.uniforms.uChaosNoise.value = THREE.MathUtils.lerp(chladniMatRef.current.uniforms.uChaosNoise.value, chaos, 0.1);
     }
  });

  const horizonShader = useMemo(() => ({
    uniforms: {
      uSunPos: { value: new THREE.Vector3() },
      uSunColor: { value: new THREE.Color() }
    },
    vertexShader: `
      varying vec3 vWorldPos;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPosition.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `,
    fragmentShader: `
      uniform vec3 uSunPos;
      uniform vec3 uSunColor;
      varying vec3 vWorldPos;
      varying vec2 vUv;
      
      void main() {
        float alpha = smoothstep(0.0, 0.5, vUv.y) * smoothstep(1.0, 0.5, vUv.y);
        vec3 dirToSun = normalize(uSunPos - vWorldPos);
        vec3 viewDir = normalize(vWorldPos);
        float sunIntensity = max(0.0, dot(viewDir, normalize(uSunPos)));
        sunIntensity = pow(sunIntensity, 4.0);
        vec3 baseColor = vec3(0.1, 0.4, 0.8);
        vec3 finalColor = mix(baseColor, uSunColor, sunIntensity * 0.8);
        alpha *= (0.3 + sunIntensity * 0.7);
        gl_FragColor = vec4(finalColor, alpha * 0.6);
      }
    `
  }), []);

  const chladniShader = useMemo(() => ({
    uniforms: {
      uTime: { value: 0 },
      uHarmonic432: { value: 0 },
      uHarmonic528: { value: 0 },
      uChaosNoise: { value: 0 }
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float uTime;
      uniform float uHarmonic432;
      uniform float uHarmonic528;
      uniform float uChaosNoise;
      varying vec2 vUv;
      
      // Chladni plate equation approximation
      float chladni(vec2 uv, float m, float n) {
        return cos(n * 3.1415 * uv.x) * cos(m * 3.1415 * uv.y) - cos(m * 3.1415 * uv.x) * cos(n * 3.1415 * uv.y);
      }

      void main() {
        vec2 centered = vUv - 0.5;
        float r = length(centered) * 2.0;
        if (r > 1.0) discard;
        
        // Base pulses (Dielectric base frequency)
        float c1 = chladni(centered * 2.0, 3.0, 5.0);
        float c2 = chladni(centered * 2.0, 4.0, 7.0);
        float c3 = chladni(centered * 2.0, 7.0, 10.0);
        
        // Let audio drive the patterns intensely
        float h432 = uHarmonic432 * 2.0;
        float h528 = uHarmonic528 * 2.0;
        float chaos = uChaosNoise * 0.5;

        // Pattern shifting based on harmonics and noise
        float val = c1 * (0.5 + h432) + c2 * (0.2 + h528) + c3 * chaos * sin(uTime * 10.0);
        
        // Add resonant rings based on distance from center (Standing waves)
        float rings = sin(r * 20.0 - uTime * 2.0 + h432 * 5.0) * 0.5 + 0.5;
        val += rings * h528 * 0.5;
        
        val = abs(val);
        float linePattern = smoothstep(0.01, 0.1, val) * smoothstep(0.4, 0.1, val);
        
        // Base color mapping: Blue base, Purple for 432Hz, Green/Gold for 528Hz, Red for chaos noise
        vec3 col = vec3(0.1, 0.4, 0.8);
        col = mix(col, vec3(0.5, 0.1, 0.9), h432); // Purple Bloom
        col = mix(col, vec3(0.2, 0.9, 0.4), h528); // Green/Gold Bloom
        col = mix(col, vec3(0.9, 0.1, 0.1), chaos); // Red Noise String

        // Base idle standing wave pulse
        float pulse = sin(uTime * 2.0 - r * 10.0) * 0.5 + 0.5;
        col = mix(col, vec3(1.0, 0.9, 0.2), pulse * 0.2); // slight idle gold pulse

        // Alpha scaling
        float alpha = linePattern * (0.3 + h432 + h528 + chaos);
        
        // Fail-safe baseline grid
        alpha = max(alpha, 0.05 * pulse);

        // Fade gracefully at edges
        alpha *= smoothstep(1.0, 0.8, r);
        
        gl_FragColor = vec4(col * (1.0 + h432 + h528), alpha);
      }
    `
  }), []);

  // Create a procedural spiral path for the sun
  const spiralPoints = useMemo(() => {
     const points = [];
     const turns = 10;
     const pointsPerTurn = 64;
     for (let i = 0; i <= turns * pointsPerTurn; i++) {
        const t = i / (turns * pointsPerTurn); // 0 to 1
        const angle = t * Math.PI * 2 * turns;
        const radius = rEquator - (rEquator - rCancer) * Math.sin(t * Math.PI * 2);
        points.push(new THREE.Vector3(Math.cos(angle) * radius, altitude, Math.sin(angle) * radius));
     }
     return points;
  }, []);
  
  const spiralGeom = useMemo(() => new THREE.BufferGeometry().setFromPoints(spiralPoints), [spiralPoints]);
  
  // Radial Azimuthal Lines (Grid-View Tracer) -> Converted to Centripetal Magnetic Vortex Shell
  const vortexLines = useMemo(() => {
     const points = [];
     const branches = 18;
     for(let i = 0; i < branches; i++) {
        const armAngleOffset = (i / branches) * Math.PI * 2;
        
        // Let's create a curve from the center (North pole) out to the edge, then underneath
        for (let r = 0; r <= 5; r += 0.25) { // 5 is the outer rim (maxR equivalent)
            // Add a spiral precession that increases with radius
            const spiralAngle = armAngleOffset + (r / 5) * Math.PI * 1.5;
            
            const x = Math.cos(spiralAngle) * r;
            const z = Math.sin(spiralAngle) * r;
            
            // To make it a segment for LineSegments, we need pairs of points
            if (r > 0) {
                 const prevR = r - 0.25;
                 const prevAngle = armAngleOffset + (prevR / 5) * Math.PI * 1.5;
                 const px = Math.cos(prevAngle) * prevR;
                 const pz = Math.sin(prevAngle) * prevR;
                 
                  // Top side connection
                 points.push(new THREE.Vector3(px, 0.05, pz));
                 points.push(new THREE.Vector3(x, 0.05, z));

                  // Bottom side wrapping (underneath the Earth)
                 points.push(new THREE.Vector3(px, -0.25, pz));
                 points.push(new THREE.Vector3(x, -0.25, z));
            }
        }
        
        // Connect the top to bottom at the very center (North pole) 
        points.push(new THREE.Vector3(0, 0.05, 0));
        points.push(new THREE.Vector3(0, -0.25, 0));

        // Connect the top to the bottom at the very edge
        const edgeAngle = armAngleOffset + Math.PI * 1.5;
        points.push(new THREE.Vector3(Math.cos(edgeAngle) * 5, 0.05, Math.sin(edgeAngle) * 5));
        points.push(new THREE.Vector3(Math.cos(edgeAngle) * 5, -0.25, Math.sin(edgeAngle) * 5));
     }
     return new THREE.BufferGeometry().setFromPoints(points);
  }, []);

  // Time uniform for vortex
  const vortexMatRef = useRef<THREE.LineBasicMaterial>(null);
  const vortexGroupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
     if (horizonMatRef.current && sunPosRef?.current && sunColorRef?.current) {
        horizonMatRef.current.uniforms.uSunPos.value.copy(sunPosRef.current);
        horizonMatRef.current.uniforms.uSunColor.value.copy(sunColorRef.current);
     }
     if (chladniMatRef.current) {
        chladniMatRef.current.uniforms.uTime.value = state.clock.elapsedTime;
     }
     if (vortexGroupRef.current) {
        vortexGroupRef.current.rotation.y = state.clock.elapsedTime * 0.1; // Rotate the whole vortex slowly
     }
  });
  return (
     <group>
        {/* Dielectric Latitudes - Tropic of Cancer */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, altitude, 0]}>
           <ringGeometry args={[rCancer - 0.01, rCancer + 0.01, 64]} />
           <meshBasicMaterial color="#ef4444" transparent opacity={0.6} side={THREE.DoubleSide} />
        </mesh>
        
        {/* Equator */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, altitude, 0]}>
           <ringGeometry args={[rEquator - 0.01, rEquator + 0.01, 64]} />
           <meshBasicMaterial color="#22c55e" transparent opacity={0.6} side={THREE.DoubleSide} />
        </mesh>
        
        {/* Tropic of Capricorn */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, altitude, 0]}>
           <ringGeometry args={[rCapricorn - 0.01, rCapricorn + 0.01, 64]} />
           <meshBasicMaterial color="#3b82f6" transparent opacity={0.6} side={THREE.DoubleSide} />
        </mesh>
        
        {/* The Solar Spiral */}
        <primitive object={new THREE.Line(spiralGeom, new THREE.LineBasicMaterial({ color: '#fef08a', transparent: true, opacity: 0.3 }))} />
        
        {/* Centripetal Magnetic Vortex Field */}
        <group ref={vortexGroupRef}>
            <primitive object={new THREE.LineSegments(vortexLines, new THREE.LineBasicMaterial({ color: '#4ade80', transparent: true, opacity: 0.25 }))} />
        </group>
        
        <gridHelper args={[10, 20, '#38bdf8', '#0284c7']} position={[0, -0.01, 0]} material-opacity={0.15} material-transparent />
        <polarGridHelper args={[5, 16, 8, 64, '#38bdf8', '#0284c7']} position={[0, -0.005, 0]} material-opacity={0.15} material-transparent />
        
        {/* Acoustic Frequency Scanner (Chladni Plate) */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
           <planeGeometry args={[10, 10]} />
           <shaderMaterial
              ref={chladniMatRef}
              transparent
              depthWrite={false}
              args={[chladniShader]}
              blending={THREE.AdditiveBlending}
           />
        </mesh>

        {/* Dynamic Horizon */}
        <mesh position={[0, 0.5, 0]}>
           <cylinderGeometry args={[5, 5, 2, 64, 1, true]} />
           <shaderMaterial
              ref={horizonMatRef}
              transparent={true}
              depthWrite={false}
              side={THREE.BackSide}
              args={[horizonShader]}
           />
        </mesh>
     </group>
  );
}

function CircuitMesh({ time }) {
    const groupRef = useRef<THREE.Group>(null);
    const travelersRef = useRef<THREE.Mesh[]>([]);

   const rCancer = 1.25;
   const rEquator = 2.5;
   const rCapricorn = 3.5;
   
   const journeyRadii = [rCancer * 0.5, rEquator * 1.1, rCapricorn * 0.9];
   const colors = ['#60a5fa', '#34d399', '#f87171'];
   
   useFrame(() => {
       if (travelersRef.current.length > 0) {
           journeyRadii.forEach((jr, idx) => {
               const groundSpeed = 0.5; 
               const angularSpeed = groundSpeed / jr; 
               // Need time.current.time here
               const t = time.current ? time.current.time : 0;
               const travelerAngle = -(t * 50 * angularSpeed);
               
               const tx = Math.cos(travelerAngle) * jr;
               const tz = Math.sin(travelerAngle) * jr;
               
               if(travelersRef.current[idx]) {
                   travelersRef.current[idx].position.set(tx, 0.1, tz);
               }
           });
       }
   });

   return (
       <group ref={groupRef}>
           {journeyRadii.map((jr, idx) => (
               <React.Fragment key={idx}>
                   {/* Orbit Path */}
                   <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
                       <ringGeometry args={[jr - 0.02, jr + 0.02, 64]} />
                       <meshBasicMaterial color={colors[idx]} transparent opacity={0.15} side={THREE.DoubleSide} />
                   </mesh>
                   {/* Traveler */}
                   <mesh ref={(el) => { if (el) travelersRef.current[idx] = el; }}>
                       <sphereGeometry args={[0.04, 16, 16]} />
                       <meshBasicMaterial color={colors[idx]} transparent opacity={0.6}/>
                   </mesh>
               </React.Fragment>
           ))}
       </group>
   );
}
