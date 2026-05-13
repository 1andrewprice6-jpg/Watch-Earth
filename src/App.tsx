import React, { useEffect, useRef, useState } from 'react';
import { Target, Battery, Navigation, Cuboid, Plane } from 'lucide-react';
import Sovereign3D from './Sovereign3D';

export default function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [realTime, setRealTime] = useState(new Date());
  const [viewMode, setViewMode] = useState<'top' | 'obs' | 'ar'>('top');
  const [showJourney, setShowJourney] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  
  const [metrics, setMetrics] = useState({
    day: 0,
    solarRadius: 0,
    lunarLag: 0,
    phase: 'New Moon',
    state: 'Equinox',
    isReset: false,
    speedMultiplier: 1.0,
  });

  const simState = useRef({
    time: (new Date().getTime() - new Date(Date.UTC(new Date().getUTCFullYear(), 0, 0)).getTime()) / 86400000, 
    speed: 1.0, 
    running: true,
    lastTime: performance.now(),
    lastStateUpdate: 0,
  });

  useEffect(() => {
    const timer = setInterval(() => setRealTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    simState.current.speed = speed;
  }, [speed]);

  // Main Render Loop
  useEffect(() => {
    let animId: number;

    const render = (now: number) => {
      const sim = simState.current;
      if (sim.running) {
        const dtSec = (now - sim.lastTime) / 1000;
        // Real-time speed: 1.0 means 1 real second = 1 real second
        // Since sim.time is in days, we divide by 86400
        sim.time += (dtSec / 86400.0) * sim.speed;
      }
      sim.lastTime = now;

      const FULL_CYCLE_DAYS = 365.25;
      const normalDays = 365;
      const yearDay = sim.time % FULL_CYCLE_DAYS;
      const isReset = false; // Disable reset to let it run properly

      if (now - sim.lastStateUpdate > 100) {
        sim.lastStateUpdate = now;
        
        const lagRad = yearDay * (Math.PI * 2 / 28);
        const rEquator = 100;
        const rCancer = 50;
        const rCapricorn = 150;
        const sunRadius = rEquator + (rCapricorn - rEquator) * Math.cos((yearDay - 355) / 365.25 * Math.PI * 2);

        let phase = 'New Moon';
        const lagMod = (lagRad % (Math.PI * 2)) / (Math.PI * 2);
        if (lagMod > 0.05 && lagMod < 0.45) phase = 'Waxing Crescent';
        else if (lagMod > 0.45 && lagMod < 0.55) phase = 'Full Moon';
        else if (lagMod > 0.55 && lagMod < 0.95) phase = 'Waning Crescent';

        const zoomRatio = (rCapricorn - sunRadius) / (rCapricorn - rCancer); 
        let stateStr = 'EQNX-CALIB';
        if (isReset) stateStr = 'DIELECTRIC DISCHARGE';
        else if (zoomRatio > 0.7) stateStr = 'SUMMER-ZOOM';
        else if (zoomRatio < 0.3) stateStr = 'WINTER-FOG';

        setMetrics(m => ({
          ...m,
          day: Math.floor(yearDay),
          solarRadius: Math.floor((sunRadius / rCapricorn) * 12400),
          lunarLag: Math.floor((lagRad % (Math.PI * 2)) * 180 / Math.PI),
          phase: phase,
          state: stateStr,
          isReset
        }));
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, []);

  const timeString = realTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
  const secondsString = realTime.toLocaleTimeString('en-US', { hour12: false, second: '2-digit' }).split(':')[2] || realTime.getSeconds().toString().padStart(2, 'Math.O');
  
  const dateString = realTime.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }).toUpperCase();

  const toggleViewMode = () => {
    setViewMode(curr => {
      if (curr === 'top') return 'obs';
      if (curr === 'obs') return 'ar';
      return 'top';
    });
  };

  const [cameraError, setCameraError] = useState('');
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (viewMode === 'ar') {
      setCameraError('');
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch(err => {
          console.error("AR Camera error:", err);
          setCameraError('Permission denied');
        });
    } else {
      setCameraError('');
      if (videoRef.current && videoRef.current.srcObject) {
         const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
         tracks.forEach(t => t.stop());
         videoRef.current.srcObject = null;
      }
    }
  }, [viewMode]);

  return (
    <div className="flex flex-col h-screen w-full bg-zinc-950 font-sans text-gray-300 items-center justify-center gap-8">
      {/* OnePlus 3 Pro Watch Face Container */}
      <div 
        onClick={toggleViewMode}
        className="relative shadow-2xl rounded-full bg-black flex items-center justify-center overflow-hidden ring-[12px] ring-zinc-800 cursor-pointer"
        style={{ width: '466px', height: '466px' }}
      >
        <div className="scanlines z-10 pointer-events-none" />

        {/* The Engine */}
        <div className="absolute inset-0 z-0" ref={containerRef}>
          {viewMode === 'ar' && (
            <>
              <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover opacity-80 pointer-events-none" />
              {cameraError && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-20">
                  <div className="text-red-400 font-mono text-xs text-center px-8 border border-red-500/30 bg-red-500/10 py-2 rounded">
                    CAMERA ACCESS DENIED
                    <br />
                    <span className="text-zinc-500 text-[10px]">Please allow camera permissions</span>
                  </div>
                </div>
              )}
            </>
          )}
          <Sovereign3D simState={simState} setMetrics={setMetrics} viewMode={viewMode} showJourney={showJourney} />
        </div>

        {/* Watch Face Foreground UI */}
        <div className="absolute inset-0 z-20 flex flex-col justify-between items-center py-10 pointer-events-none">
          
          {viewMode === 'ar' && (
             <div className="absolute top-1/3 right-6 transform -translate-y-1/2 flex flex-col items-end pointer-events-none">
                <div className="font-mono text-[8px] text-emerald-400/80 bg-emerald-950/40 px-1 rounded-sm border border-emerald-900/50 mb-1">SPECTRAL CALIBRATION ON</div>
                <div className="font-mono text-[8px] text-zinc-500">LUX DENSITY</div>
                <div className="font-mono text-xl font-bold text-amber-400 drop-shadow-[0_0_8px_rgba(251,191,36,0.6)]">
                   {metrics?.zoomRatio ? (metrics.zoomRatio * 1420 + 200).toFixed(0) : 1620} <span className="text-[10px] text-zinc-500">lx</span>
                </div>
                <div className="font-mono text-[8px] text-zinc-500 mt-1">REFRACTIVE LOSS</div>
                <div className="font-mono text-[10px] text-red-400">
                   {metrics?.zoomRatio ? ((1.0 - metrics.zoomRatio) * 64.2).toFixed(1) : 0}%
                </div>
             </div>
          )}

          {/* Removed AR Target Focus Info here */}

          {/* Center North Pole: Kernel Root-State */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center pointer-events-none z-10">
             <div className="w-2 h-2 rounded-full border border-emerald-500/50 flex items-center justify-center">
                <div className="w-1 h-1 bg-emerald-400 rounded-full animate-pulse shadow-[0_0_8px_4px_rgba(52,211,153,0.4)]" />
             </div>
             <div className="mt-1 text-[6px] font-mono tracking-widest text-emerald-400/80 bg-black/40 px-1 rounded-sm border border-emerald-900/30">ROOT_SECURE</div>
          </div>

          {/* Top Arc: Status & Date */}
          <div className="flex flex-col items-center space-y-1 z-50 pointer-events-auto bg-black/0 backdrop-blur-[0px] px-6 py-2 rounded-full border border-white/0 hover:bg-black/5 hover:backdrop-blur-sm transition-all">
             <div className="flex items-center gap-3 text-zinc-300 font-bold tracking-widest text-[10px]">
               <div className="flex items-center gap-1">
                 <Battery className="w-3 h-3 text-emerald-400" />
                 100%
               </div>
               <div className="h-3 w-px bg-white/20"></div>
               <button 
                 onClick={(e) => { e.stopPropagation(); setShowJourney(!showJourney); }}
                 className={`flex items-center gap-1 px-2 py-0.5 rounded transition-all border ${showJourney ? 'text-blue-300 border-blue-400/30 bg-blue-500/10 shadow-[0_0_10px_rgba(59,130,246,0.3)]' : 'text-zinc-400 border-transparent hover:text-white'}`}
               >
                 <Plane className="w-3 h-3" />
                 CIRCUIT
               </button>
             </div>
             <div className="text-white font-bold tracking-widest text-xs uppercase drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)]">
                {dateString}
             </div>
          </div>

          {/* Center: Time */}
          <div className="flex flex-col items-center justify-center bg-transparent px-10 py-5 pointer-events-none">
             <div 
               className="font-mono text-7xl md:text-8xl font-black tracking-tighter flex items-baseline select-none drop-shadow-[0_4px_8px_rgba(0,0,0,0.5)]"
               style={{ WebkitTextStroke: '1px rgba(255,255,255,0.8)', color: 'rgba(255,255,255,0.05)' }}
             >
                {timeString.split(':')[0]}
                <span 
                  className="mx-2 text-6xl md:text-7xl"
                  style={{ WebkitTextStroke: `1px ${metrics.isReset ? 'rgba(52,211,153,0.8)' : 'rgba(245,158,11,0.8)'}`, color: 'rgba(255,255,255,0.05)' }}
                >:</span>
                {timeString.split(':')[1]}
             </div>
             <div className="flex items-center justify-center gap-3 mt-3 opacity-70">
               <span className={`font-mono text-[10px] md:text-xs tracking-widest px-3 py-1 rounded shadow-inner transition-colors ${metrics.isReset ? 'text-emerald-300 bg-emerald-900/30 border border-emerald-500/30' : 'text-zinc-200 bg-black/30 border border-white/10'}`}>
                 {viewMode === 'top' ? 'SOV-13 (3D-TOP)' : viewMode === 'obs' ? 'SOV-13 (3D-OBS)' : 'FIRMAMENT APERTURE (AR)'}
               </span>
               <span className="font-mono text-sm md:text-base tracking-widest text-white font-bold bg-black/30 shadow-inner px-3 py-1 rounded border border-white/10">
                 {secondsString}
               </span>
             </div>
          </div>

          {/* Bottom Arc: Simulation Data */}
          <div className="flex flex-col items-center bg-black/0 backdrop-blur-[0px] px-6 py-3 rounded-full border border-white/0 hover:bg-black/5 hover:backdrop-blur-sm transition-all space-y-2 pointer-events-auto">
            <div className={`px-3 py-1 rounded text-[10px] font-mono tracking-widest font-bold shadow-inner border ${metrics.isReset ? 'text-emerald-300 bg-emerald-900/10 border-emerald-500/30 shadow-[0_0_8px_rgba(52,211,153,0.1)]' : 'text-zinc-200 bg-black/10 border-white/10'}`}>
              {metrics.state}
            </div>
            <div className="flex gap-6 text-[10px] font-mono tracking-widest text-white font-semibold flex-wrap justify-center">
               <div className="flex items-center gap-1.5 drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)]">
                 <Target className="w-3.5 h-3.5 text-amber-500" />
                 T-{metrics.day.toString().padStart(3, '0')}
               </div>
               <div className="w-px h-3 bg-white/20 self-center"></div>
               <div className="flex items-center gap-1.5 drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)]">
                 <Navigation className="w-3.5 h-3.5 text-blue-400 transform transition-transform" style={{ rotate: `${metrics.lunarLag}deg` }} />
                 {metrics.phase}
               </div>
            </div>
          </div>

        </div>
        
        {/* Ambient Display Vignette */}
        <div className="absolute inset-0 pointer-events-none rounded-full z-30" />
      </div>

      {/* Speed Control Slider */}
      <div className="flex flex-col items-center space-y-2 w-72 bg-transparent backdrop-blur-[2px] p-5 rounded-3xl shadow-[0_8px_32px_rgba(0,0,0,0.2)] border border-white/5 hover:bg-black/40 hover:backdrop-blur-md transition-all mt-6 md:mt-8 z-50">
        <label className="text-xs tracking-widest font-mono text-white font-bold drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)]">
          SIM SPEED: <span className="text-emerald-400">{speed.toFixed(1)}x</span>
        </label>
        <input 
          type="range"
          min="1.0"
          max="1000.0"
          step="1.0"
          value={speed}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
          className="w-full accent-emerald-500 h-1.5 bg-zinc-800 rounded-full appearance-none outline-none focus:ring-2 focus:ring-emerald-500/50 cursor-pointer opacity-70 hover:opacity-100 transition-opacity"
        />
        <div className="w-full flex justify-between text-[8px] font-mono text-zinc-500 font-bold px-1 mt-1">
          <span>REALTIME</span>
          <span>1000x</span>
        </div>
      </div>

      <button 
        onClick={toggleViewMode}
        className="font-mono text-xs font-bold tracking-widest text-zinc-300 hover:text-white px-8 py-3 bg-transparent backdrop-blur-[2px] border border-white/5 hover:bg-black/40 hover:backdrop-blur-md hover:border-white/30 rounded-full shadow-[0_8px_32px_rgba(0,0,0,0.2)] transition-all mt-4 z-50 mb-10 md:mb-0"
      >
        CYCLE VIEW MODE <span className="text-emerald-400 ml-1">({viewMode.toUpperCase()})</span>
      </button>
    </div>
  );
}

