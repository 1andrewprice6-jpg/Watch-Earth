import * as THREE from 'three';

export function createMagneticLeyLines(numLines = 24, numPoints = 64) {
  const lines = [];
  const radius = 5.0; 
  
  for (let i = 0; i < numLines; i++) {
    const angle = (i / numLines) * Math.PI * 2;
    const points: THREE.Vector3[] = [];
    
    // A spiraling arch
    for (let j = 0; j <= numPoints; j++) {
      const t = j / numPoints; // 0 to 1
      
      const r = t * radius;
      // height arcs up and then down -> sin(t * pi)
      const h = Math.sin(t * Math.PI) * 1.5; 
      
      // Let it spiral slightly
      const spiralOffset = t * Math.PI * 0.5;
      
      const x = Math.cos(angle + spiralOffset) * r;
      const z = Math.sin(angle + spiralOffset) * r;
      
      points.push(new THREE.Vector3(x, h, z));
    }
    lines.push(points);
  }
  return lines;
}
