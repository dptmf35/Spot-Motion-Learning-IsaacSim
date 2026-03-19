import { useState } from 'react'
import { useGaitWebSocket } from './hooks/useGaitWebSocket'
import GaitDataCollection from './tabs/GaitDataCollection'

export default function App() {
  const { connected } = useGaitWebSocket()

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="header-logo">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect x="4" y="10" width="20" height="12" rx="3" stroke="#B4D3D9" strokeWidth="1.5" />
              <circle cx="10" cy="16" r="2.5" fill="#B4D3D9" />
              <circle cx="18" cy="16" r="2.5" fill="#B4D3D9" />
              <path d="M10 10V7" stroke="#B4D3D9" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M18 10V7" stroke="#B4D3D9" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M14 10V6" stroke="#B4D3D9" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M4 19H2" stroke="#B4D3D9" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M26 19H24" stroke="#B4D3D9" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M6 22H4v2" stroke="#B4D3D9" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M22 22H24v2" stroke="#B4D3D9" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <div className="header-title">Spot Gait Data Collection</div>
            <div className="header-sub">IsaacSim 5.0 — Autonomous Navigation</div>
          </div>
        </div>

        <div className="header-right">
          <div className={`conn-status ${connected ? 'conn-status--on' : 'conn-status--off'}`}>
            <span className="conn-dot" />
            <span className="conn-label">{connected ? 'Live' : 'Offline'}</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="main">
        <GaitDataCollection />
      </main>
    </div>
  )
}
