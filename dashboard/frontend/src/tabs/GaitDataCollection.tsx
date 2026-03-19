import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import './GaitDataCollection.css'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
} from 'recharts'
import { useGaitWebSocket, GaitWsPayload } from '../hooks/useGaitWebSocket'
import { useGaitEpisodes, useCollectionControls, CollectionConfig } from '../hooks/useGaitApi'

// ── Joint metadata ──────────────────────────────────────────────────────────

const JOINT_NAMES = [
  'FL_hip_x', 'FL_hip_y', 'FL_knee',
  'FR_hip_x', 'FR_hip_y', 'FR_knee',
  'RL_hip_x', 'RL_hip_y', 'RL_knee',
  'RR_hip_x', 'RR_hip_y', 'RR_knee',
]

const LEG_COLORS: Record<string, string> = {
  FL: '#7AABB3',
  FR: '#9B8EC7',
  RL: '#BDA6CE',
  RR: '#6B5F7A',
}

const FOOT_LABELS = ['FL', 'FR', 'RL', 'RR']

const FOOT_COLORS = [
  '#B4D3D9',  // FL: teal
  '#9B8EC7',  // FR: purple
  '#BDA6CE',  // RL: lavender
  '#7AABB3',  // RR: dark teal
]

function jointStrokeDash(name: string): string {
  if (name.endsWith('_knee')) return '4 3'
  if (name.endsWith('_hip_y')) return '8 3'
  return '0'
}

function jointColor(name: string): string {
  const leg = Object.keys(LEG_COLORS).find(l => name.startsWith(l))
  return leg ? LEG_COLORS[leg] : '#A496B0'
}

// ── Tooltip style shared ─────────────────────────────────────────────────────

const TOOLTIP_STYLE = {
  contentStyle: {
    background: '#FDFAF7',
    border: '1px solid #B4D3D9',
    borderRadius: '8px',
    fontSize: '11px',
    boxShadow: '0 4px 12px rgba(45,38,64,0.1)',
  },
  labelStyle: { color: '#6B5F7A' },
  itemStyle: { color: '#2D2640' },
}

// ── Utilities ────────────────────────────────────────────────────────────────

function safeNum(v: number): number {
  return isFinite(v) ? v : 0
}

function formatDuration(secs: number): string {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  const ds = Math.floor((secs % 1) * 10)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ds}`
}

function formatTime(isoStr: string): string {
  try {
    return new Date(isoStr).toLocaleTimeString('en-US', { hour12: false })
  } catch {
    return isoStr
  }
}

// ── Convex hull (gift wrapping) ───────────────────────────────────────────────

function cross(O: [number, number], A: [number, number], B: [number, number]): number {
  return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
}

function convexHull(points: [number, number][]): [number, number][] {
  const n = points.length
  if (n < 2) return points
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const lower: [number, number][] = []
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop()
    }
    lower.push(p)
  }
  const upper: [number, number][] = []
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop()
    }
    upper.push(p)
  }
  upper.pop()
  lower.pop()
  return [...lower, ...upper]
}

// ── Canvas Top-Down Trajectory ───────────────────────────────────────────────

interface CanvasTrajectoryProps {
  state: GaitWsPayload | null
  comTrail: Array<{ x: number; y: number }>
  zmpTrail: Array<{ x: number; y: number }>
}

// Oblique projection: X axis goes right, Y axis goes up-right (shear) + compressed
// Creates a 3D floor-tile perspective feel
const ISO_SHEAR = 0.42   // X offset per world Y unit (right-lean of Y axis)
const ISO_YSCALE = 0.52  // Y compression ratio — strong enough to look 3D

function CanvasTrajectory({ state, comTrail, zmpTrail }: CanvasTrajectoryProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)

  const stateRef = useRef(state)
  const comTrailRef = useRef(comTrail)
  const zmpTrailRef = useRef(zmpTrail)

  useEffect(() => { stateRef.current = state }, [state])
  useEffect(() => { comTrailRef.current = comTrail }, [comTrail])
  useEffect(() => { zmpTrailRef.current = zmpTrail }, [zmpTrail])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    function draw() {
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const dpr = window.devicePixelRatio || 1
      const W = canvas.clientWidth
      const H = canvas.clientHeight

      if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
        canvas.width = W * dpr
        canvas.height = H * dpr
        ctx.scale(dpr, dpr)
      }

      const s = stateRef.current
      const comTrailData = comTrailRef.current
      const zmpTrailData = zmpTrailRef.current

      // Oblique projection: world (wx, wy) → iso (ix, iy)
      // Y axis is sheared right and compressed to create 3D floor look
      function isoProj(wx: number, wy: number): [number, number] {
        return [wx - wy * ISO_SHEAR, wy * ISO_YSCALE]
      }

      // Gather all projected points to determine viewport
      const allPts: [number, number][] = []
      if (s) {
        allPts.push(isoProj(safeNum(s.com_position[0]), safeNum(s.com_position[1])))
        allPts.push(isoProj(safeNum(s.zmp_position[0]), safeNum(s.zmp_position[1])))
        s.foot_positions.forEach(fp => allPts.push(isoProj(safeNum(fp[0]), safeNum(fp[1]))))
        if (s.collection?.current_waypoint) {
          allPts.push(isoProj(safeNum(s.collection.current_waypoint[0]), safeNum(s.collection.current_waypoint[1])))
        }
      }
      comTrailData.forEach(p => allPts.push(isoProj(p.x, p.y)))

      const pad = 1.5
      let ixMin = -2, ixMax = 2, iyMin = -1.2, iyMax = 1.2
      if (allPts.length > 0) {
        ixMin = Math.min(...allPts.map(p => p[0])) - pad
        ixMax = Math.max(...allPts.map(p => p[0])) + pad
        iyMin = Math.min(...allPts.map(p => p[1])) - pad * ISO_YSCALE
        iyMax = Math.max(...allPts.map(p => p[1])) + pad * ISO_YSCALE
      }

      const isoW = ixMax - ixMin
      const isoH = iyMax - iyMin
      const aspect = isoW / isoH
      const canvasAspect = W / H
      let scale: number, offX: number, offY: number
      if (aspect > canvasAspect) {
        scale = W / isoW
        offX = 0
        offY = (H - isoH * scale) / 2
      } else {
        scale = H / isoH
        offX = (W - isoW * scale) / 2
        offY = 0
      }

      // World → canvas (flips Y because canvas Y increases downward)
      function toCanvas(wx: number, wy: number): [number, number] {
        const [ix, iy] = isoProj(wx, wy)
        return [offX + (ix - ixMin) * scale, offY + (iyMax - iy) * scale]
      }

      // ── 1. Background: depth gradient (darker = far = top) ─────────
      const bgGrad = ctx.createLinearGradient(0, 0, 0, H)
      bgGrad.addColorStop(0, '#E8E5DF')
      bgGrad.addColorStop(1, '#F8F6F2')
      ctx.fillStyle = bgGrad
      ctx.fillRect(0, 0, W, H)

      // ── 2. Oblique floor grid (parallelogram tiles → 3D floor look) ─
      // Compute world bounds that cover the viewport generously
      const wyWorldRange = (iyMax - iyMin) / ISO_YSCALE + 4
      const wyWorldMin = -wyWorldRange / 2
      const wyWorldMax = wyWorldRange / 2
      const wxWorldMin = ixMin - 2
      const wxWorldMax = ixMax + 2

      const gridStep = 1.0

      // World Y = const lines → tilted due to shear (look like floor rows receding)
      ctx.lineWidth = 0.5
      for (let wy = Math.ceil(wyWorldMin); wy <= wyWorldMax; wy += gridStep) {
        const prog = (wy - wyWorldMin) / (wyWorldMax - wyWorldMin)
        const alpha = 0.04 + 0.11 * (1 - prog)  // far = darker
        ctx.strokeStyle = `rgba(0,0,0,${alpha.toFixed(3)})`
        const [x0, y0] = toCanvas(wxWorldMin, wy)
        const [x1, y1] = toCanvas(wxWorldMax, wy)
        ctx.beginPath()
        ctx.moveTo(x0, y0)
        ctx.lineTo(x1, y1)
        ctx.stroke()
      }

      // World X = const lines → diagonal lines cutting across tiles
      for (let wx = Math.ceil(wxWorldMin); wx <= wxWorldMax; wx += gridStep) {
        ctx.strokeStyle = 'rgba(0,0,0,0.055)'
        const [x0, y0] = toCanvas(wx, wyWorldMin)
        const [x1, y1] = toCanvas(wx, wyWorldMax)
        ctx.beginPath()
        ctx.moveTo(x0, y0)
        ctx.lineTo(x1, y1)
        ctx.stroke()
      }

      // ── 3. CoM trail ──────────────────────────────────────────────
      if (comTrailData.length > 1) {
        for (let i = 1; i < comTrailData.length; i++) {
          const alpha = 0.06 + 0.55 * (i / comTrailData.length)
          const [x0, y0] = toCanvas(comTrailData[i - 1].x, comTrailData[i - 1].y)
          const [x1, y1] = toCanvas(comTrailData[i].x, comTrailData[i].y)
          ctx.beginPath()
          ctx.strokeStyle = `rgba(99,153,34,${alpha})`
          ctx.lineWidth = 1.5
          ctx.moveTo(x0, y0)
          ctx.lineTo(x1, y1)
          ctx.stroke()
        }
      }

      // ── 4. Support polygon ─────────────────────────────────────────
      if (s) {
        const contactFeet: [number, number][] = s.foot_positions
          .map((fp, i) => ({ fp, contact: s.foot_contact[i] }))
          .filter(d => d.contact)
          .map(d => [safeNum(d.fp[0]), safeNum(d.fp[1])] as [number, number])
        if (contactFeet.length >= 3) {
          const hull = convexHull(contactFeet)
          if (hull.length >= 2) {
            ctx.beginPath()
            const [hx0, hy0] = toCanvas(hull[0][0], hull[0][1])
            ctx.moveTo(hx0, hy0)
            for (let i = 1; i < hull.length; i++) {
              const [hx, hy] = toCanvas(hull[i][0], hull[i][1])
              ctx.lineTo(hx, hy)
            }
            ctx.closePath()
            ctx.fillStyle = 'rgba(29,158,117,0.14)'
            ctx.fill()
            ctx.strokeStyle = '#0F9E6E'
            ctx.lineWidth = 1.2
            ctx.stroke()
          }
        } else if (contactFeet.length === 2) {
          const [ax, ay] = toCanvas(contactFeet[0][0], contactFeet[0][1])
          const [bx, by] = toCanvas(contactFeet[1][0], contactFeet[1][1])
          ctx.beginPath()
          ctx.moveTo(ax, ay)
          ctx.lineTo(bx, by)
          ctx.strokeStyle = '#0F9E6E'
          ctx.lineWidth = 1.2
          ctx.stroke()
        }
      }

      // ── 5. Foot positions with elliptical ground shadow ────────────
      if (s) {
        s.foot_positions.forEach((fp, i) => {
          const wx = safeNum(fp[0])
          const wy = safeNum(fp[1])
          const [cx, cy] = toCanvas(wx, wy)
          const inContact = s.foot_contact[i]
          const r = inContact ? 7 : 5
          const color = inContact ? '#D85A30' : '#F0997B'

          // Elliptical ground shadow (offset down-right, squashed for floor feel)
          ctx.beginPath()
          ctx.ellipse(cx + 3, cy + 4, r + 3, (r + 1) * 0.45, 0, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(0,0,0,0.10)'
          ctx.fill()

          // Foot dot with glow
          ctx.shadowColor = inContact ? 'rgba(216,90,48,0.45)' : 'transparent'
          ctx.shadowBlur = inContact ? 9 : 0
          ctx.beginPath()
          ctx.arc(cx, cy, r, 0, Math.PI * 2)
          if (inContact) {
            ctx.fillStyle = color
            ctx.fill()
          } else {
            ctx.strokeStyle = color
            ctx.lineWidth = 1.5
            ctx.stroke()
          }
          ctx.shadowBlur = 0
          ctx.shadowColor = 'transparent'

          // Foot label
          ctx.font = '9px monospace'
          ctx.fillStyle = '#C04A1A'
          ctx.fillText(FOOT_LABELS[i], cx + r + 2, cy + 3)
        })
      }

      // ── 6. ZMP trail ──────────────────────────────────────────────
      const zmpWindow = zmpTrailData.slice(-30)
      zmpWindow.forEach((p, i) => {
        const alpha = 0.15 + 0.55 * (i / zmpWindow.length)
        const [cx, cy] = toCanvas(p.x, p.y)
        ctx.beginPath()
        ctx.arc(cx, cy, 2.5, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(55,138,221,${(alpha * 0.6).toFixed(2)})`
        ctx.fill()
      })

      // ── 7. ZMP current ─────────────────────────────────────────────
      if (s) {
        const [zx, zy] = s.zmp_position
        if (isFinite(zx) && isFinite(zy)) {
          const [cx, cy] = toCanvas(zx, zy)

          // CoM→ZMP connector (draw first, underneath)
          if (isFinite(s.com_position[0]) && isFinite(s.com_position[1])) {
            const [comCx, comCy] = toCanvas(safeNum(s.com_position[0]), safeNum(s.com_position[1]))
            ctx.beginPath()
            ctx.setLineDash([3, 3])
            ctx.strokeStyle = 'rgba(0,0,0,0.20)'
            ctx.lineWidth = 0.8
            ctx.moveTo(comCx, comCy)
            ctx.lineTo(cx, cy)
            ctx.stroke()
            ctx.setLineDash([])
          }

          // Ground shadow
          ctx.beginPath()
          ctx.ellipse(cx + 2, cy + 3, 9, 4, 0, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(55,138,221,0.13)'
          ctx.fill()

          // ZMP dot
          ctx.shadowColor = 'rgba(55,138,221,0.55)'
          ctx.shadowBlur = 11
          ctx.beginPath()
          ctx.arc(cx, cy, 6, 0, Math.PI * 2)
          ctx.fillStyle = '#378ADD'
          ctx.fill()
          ctx.shadowBlur = 0
          ctx.shadowColor = 'transparent'
          ctx.strokeStyle = '#FFFFFF'
          ctx.lineWidth = 1.5
          ctx.stroke()

          ctx.font = '9px monospace'
          ctx.fillStyle = '#1560B7'
          ctx.fillText('ZMP', cx + 9, cy - 4)
        }
      }

      // ── 8. CoM current ────────────────────────────────────────────
      if (s) {
        const [comX, comY] = s.com_position
        if (isFinite(comX) && isFinite(comY)) {
          const [cx, cy] = toCanvas(safeNum(comX), safeNum(comY))

          // Ground shadow
          ctx.beginPath()
          ctx.ellipse(cx + 3, cy + 4, 12, 5, 0, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(99,153,34,0.14)'
          ctx.fill()

          // CoM dot
          ctx.shadowColor = 'rgba(99,153,34,0.55)'
          ctx.shadowBlur = 13
          ctx.beginPath()
          ctx.arc(cx, cy, 7, 0, Math.PI * 2)
          ctx.fillStyle = '#639922'
          ctx.fill()
          ctx.shadowBlur = 0
          ctx.shadowColor = 'transparent'
          ctx.strokeStyle = '#FFFFFF'
          ctx.lineWidth = 1.5
          ctx.stroke()
          ctx.font = 'bold 9px monospace'
          ctx.fillStyle = '#3A5A14'
          ctx.fillText('CoM', cx + 9, cy - 4)
        }
      }

      // ── 9. Waypoint ───────────────────────────────────────────────
      if (s?.collection?.current_waypoint) {
        const wp = s.collection.current_waypoint
        const [cx, cy] = toCanvas(safeNum(wp[0]), safeNum(wp[1]))
        const size = 9

        // Ground shadow
        ctx.beginPath()
        ctx.ellipse(cx + 3, cy + 4, size + 3, (size + 1) * 0.45, 0, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(123,107,170,0.15)'
        ctx.fill()

        // Diamond
        ctx.shadowColor = 'rgba(123,107,170,0.45)'
        ctx.shadowBlur = 9
        ctx.beginPath()
        ctx.moveTo(cx, cy - size)
        ctx.lineTo(cx + size, cy)
        ctx.lineTo(cx, cy + size)
        ctx.lineTo(cx - size, cy)
        ctx.closePath()
        ctx.fillStyle = '#7B6BAA'
        ctx.fill()
        ctx.shadowBlur = 0
        ctx.shadowColor = 'transparent'

        ctx.beginPath()
        ctx.arc(cx, cy, size + 5, 0, Math.PI * 2)
        ctx.setLineDash([2, 3])
        ctx.strokeStyle = 'rgba(123,107,170,0.5)'
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.setLineDash([])

        ctx.font = '9px monospace'
        ctx.fillStyle = '#5A4A8A'
        ctx.fillText('WP', cx + size + 4, cy + 3)
      }

      // ── Scale bar ─────────────────────────────────────────────────
      const scaleBarPx = 1.0 * scale
      const sbX = W - scaleBarPx - 12
      const sbY = H - 14
      ctx.beginPath()
      ctx.strokeStyle = 'rgba(107,95,122,0.5)'
      ctx.lineWidth = 1.5
      ctx.moveTo(sbX, sbY)
      ctx.lineTo(sbX + scaleBarPx, sbY)
      ctx.moveTo(sbX, sbY - 4)
      ctx.lineTo(sbX, sbY + 4)
      ctx.moveTo(sbX + scaleBarPx, sbY - 4)
      ctx.lineTo(sbX + scaleBarPx, sbY + 4)
      ctx.stroke()
      ctx.font = '9px monospace'
      ctx.fillStyle = 'rgba(107,95,122,0.7)'
      ctx.textAlign = 'center'
      ctx.fillText('1m', sbX + scaleBarPx / 2, sbY - 6)
      ctx.textAlign = 'left'

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(rafRef.current)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="traj-canvas"
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  )
}

// ── ZMP Time Series ──────────────────────────────────────────────────────────

interface ZmpTimeSeriesProps {
  history: GaitWsPayload[]
}

function ZmpTimeSeries({ history }: ZmpTimeSeriesProps) {
  const data = useMemo(() => {
    if (history.length === 0) return []
    const latest = history[history.length - 1].timestamp
    return history.map(s => ({
      t: -(latest - s.timestamp),
      x: safeNum(s.zmp_position[0]),
      y: safeNum(s.zmp_position[1]),
    }))
  }, [history])

  return (
    <ResponsiveContainer width="100%" height={100}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="2 3" stroke="rgba(180,211,217,0.25)" />
        <XAxis
          dataKey="t"
          type="number"
          domain={['auto', 0]}
          tickFormatter={(v: number) => `${v.toFixed(0)}s`}
          tick={{ fill: '#A496B0', fontSize: 9 }}
          stroke="rgba(180,211,217,0.3)"
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: '#A496B0', fontSize: 9 }}
          stroke="rgba(180,211,217,0.3)"
          width={32}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <Line
          type="monotone"
          dataKey="x"
          name="ZMP x"
          stroke="#378ADD"
          strokeWidth={1.2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="y"
          name="ZMP y"
          stroke="#5DCAA5"
          strokeWidth={1.2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ── CoM Time Series ──────────────────────────────────────────────────────────

interface ComTimeSeriesProps {
  history: GaitWsPayload[]
}

function ComTimeSeries({ history }: ComTimeSeriesProps) {
  const data = useMemo(() => {
    if (history.length === 0) return []
    const latest = history[history.length - 1].timestamp
    return history.map(s => ({
      t: -(latest - s.timestamp),
      x: safeNum(s.com_position[0]),
      y: safeNum(s.com_position[1]),
      z: safeNum(s.com_position[2]),
    }))
  }, [history])

  return (
    <ResponsiveContainer width="100%" height={100}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="2 3" stroke="rgba(180,211,217,0.25)" />
        <XAxis
          dataKey="t"
          type="number"
          domain={['auto', 0]}
          tickFormatter={(v: number) => `${v.toFixed(0)}s`}
          tick={{ fill: '#A496B0', fontSize: 9 }}
          stroke="rgba(180,211,217,0.3)"
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: '#A496B0', fontSize: 9 }}
          stroke="rgba(180,211,217,0.3)"
          width={32}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <Line
          type="monotone"
          dataKey="x"
          name="CoM x"
          stroke="#7AABB3"
          strokeWidth={1.2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="y"
          name="CoM y"
          stroke="#9B8EC7"
          strokeWidth={1.2}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="z"
          name="CoM z"
          stroke="#BDA6CE"
          strokeWidth={1.2}
          strokeDasharray="4 2"
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ── Navigation Trajectory (episode-level bird's eye) ────────────────────────

interface NavTrajectoryProps {
  comTrail: Array<{ x: number; y: number }>
  state: GaitWsPayload | null
}

function NavTrajectoryChart({ comTrail, state }: NavTrajectoryProps) {
  const waypoint = state?.collection?.current_waypoint ?? null

  const trailData = useMemo(() => comTrail, [comTrail])
  const waypointData = useMemo(
    () => waypoint ? [{ x: safeNum(waypoint[0]), y: safeNum(waypoint[1]) }] : [],
    [waypoint]
  )
  const robotPos = useMemo(() => {
    if (!state) return []
    return [{ x: safeNum(state.com_position[0]), y: safeNum(state.com_position[1]) }]
  }, [state])

  const allX = [
    ...trailData.map(p => p.x),
    ...waypointData.map(p => p.x),
    ...robotPos.map(p => p.x),
  ].filter(isFinite)
  const allY = [
    ...trailData.map(p => p.y),
    ...waypointData.map(p => p.y),
    ...robotPos.map(p => p.y),
  ].filter(isFinite)

  const pad = 1.5
  const xMin = allX.length ? Math.min(...allX) - pad : -5
  const xMax = allX.length ? Math.max(...allX) + pad : 5
  const yMin = allY.length ? Math.min(...allY) - pad : -5
  const yMax = allY.length ? Math.max(...allY) + pad : 5

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(180,211,217,0.25)" />
        <XAxis
          dataKey="x"
          type="number"
          domain={[xMin, xMax]}
          name="X"
          unit="m"
          tick={{ fill: '#A496B0', fontSize: 10 }}
          stroke="rgba(180,211,217,0.4)"
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <YAxis
          dataKey="y"
          type="number"
          domain={[yMin, yMax]}
          name="Y"
          unit="m"
          width={38}
          tick={{ fill: '#A496B0', fontSize: 10 }}
          stroke="rgba(180,211,217,0.4)"
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <Tooltip
          {...TOOLTIP_STYLE}
          cursor={false}
          formatter={(v: number) => v.toFixed(3)}
        />

        {/* Full episode CoM path */}
        <Scatter
          name="Episode Path"
          data={trailData}
          fill="rgba(122,171,179,0.4)"
          line={{ stroke: '#7AABB3', strokeWidth: 1, opacity: 0.7 }}
          lineType="joint"
          shape={<circle r={1} />}
          isAnimationActive={false}
        />

        {/* Current waypoint */}
        <Scatter
          name="Waypoint"
          data={waypointData}
          isAnimationActive={false}
          shape={(props: { cx?: number; cy?: number }) => {
            const { cx = 0, cy = 0 } = props
            const s = 7
            return (
              <polygon
                points={`${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`}
                fill="#7B6BAA"
                stroke="#2D2640"
                strokeWidth={1}
              />
            )
          }}
        />

        {/* Robot current position */}
        <Scatter
          name="Robot"
          data={robotPos}
          isAnimationActive={false}
          shape={(props: { cx?: number; cy?: number }) => {
            const { cx = 0, cy = 0 } = props
            return (
              <g>
                <circle cx={cx} cy={cy} r={7} fill="#639922" stroke="#FFFFFF" strokeWidth={1.5} />
              </g>
            )
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

// ── Sub-components ───────────────────────────────────────────────────────────

interface ConnectionBannerProps {
  connected: boolean
}

function ConnectionBanner({ connected }: ConnectionBannerProps) {
  if (connected) return null
  return (
    <div className="gait-offline-banner">
      <span className="gait-offline-dot" />
      Gait backend offline — connect to port 8001
    </div>
  )
}

interface ProgressBarProps {
  current: number
  total: number
}

function ProgressBar({ current, total }: ProgressBarProps) {
  const pct = total > 0 ? Math.min((current / total) * 100, 100) : 0
  return (
    <div className="gait-progress-track">
      <div className="gait-progress-fill" style={{ width: `${pct}%` }} />
    </div>
  )
}

interface JointChartProps {
  history: GaitWsPayload[]
  showVel: boolean
}

function JointChart({ history, showVel }: JointChartProps) {
  const chartData = useMemo(() => {
    if (history.length === 0) return []
    const latest = history[history.length - 1].timestamp
    return history.map(s => {
      const row: Record<string, number> = { t: -(latest - s.timestamp) }
      const arr = showVel ? s.joint_vel : s.joint_pos
      arr.forEach((v, i) => { row[`j${i}`] = safeNum(v) })
      return row
    })
  }, [history, showVel])

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(180,211,217,0.3)" />
        <XAxis
          dataKey="t"
          type="number"
          domain={['auto', 0]}
          tickFormatter={(v: number) => `${v.toFixed(1)}s`}
          tick={{ fill: '#6B5F7A', fontSize: 10 }}
          stroke="rgba(180,211,217,0.4)"
        />
        <YAxis
          tick={{ fill: '#6B5F7A', fontSize: 10 }}
          stroke="rgba(180,211,217,0.4)"
          width={38}
          tickFormatter={(v: number) => v.toFixed(2)}
        />
        <Tooltip
          {...TOOLTIP_STYLE}
          formatter={(v: number) => v.toFixed(4)}
          labelFormatter={(v: number) => `${v.toFixed(2)}s`}
        />
        <ReferenceLine y={0} stroke="rgba(45,38,64,0.12)" strokeDasharray="3 2" />
        {JOINT_NAMES.map((name, i) => (
          <Line
            key={name}
            type="monotone"
            dataKey={`j${i}`}
            name={name}
            stroke={jointColor(name)}
            strokeWidth={1.2}
            strokeDasharray={jointStrokeDash(name)}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}


// ── Stats helper types ────────────────────────────────────────────────────────

type StatLevel = 'ok' | 'warn' | 'err' | 'neutral'

function statClass(level: StatLevel): string {
  if (level === 'ok') return 'gait-stat-val gait-stat-val--ok'
  if (level === 'warn') return 'gait-stat-val gait-stat-val--warn'
  if (level === 'err') return 'gait-stat-val gait-stat-val--err'
  return 'gait-stat-val'
}

// ── ZMP Stats Panel ───────────────────────────────────────────────────────────

interface ZmpStatsPanelProps {
  state: GaitWsPayload | null
  history: GaitWsPayload[]
}

// null-safe number: returns null if value is null/undefined/NaN/Infinity
function validNum(v: unknown): number | null {
  if (v === null || v === undefined) return null
  const n = Number(v)
  return isFinite(n) ? n : null
}

function ZmpStatsPanel({ state, history }: ZmpStatsPanelProps) {
  const stats = useMemo(() => {
    if (!state) return null

    // ZMP can be null when robot is airborne — treat null as unavailable
    const zx = validNum(state.zmp_position[0])
    const zy = validNum(state.zmp_position[1])
    const zmpAvail = zx !== null && zy !== null

    // Polygon center: centroid of contact feet (use actual world positions)
    const contactFeet = state.foot_positions
      .map((fp, i) => ({ fp, contact: state.foot_contact[i] }))
      .filter(d => d.contact)
      .map(d => d.fp)

    let polyCx: number | null = null
    let polyCy: number | null = null
    if (contactFeet.length > 0) {
      polyCx = contactFeet.reduce((s, fp) => s + safeNum(fp[0]), 0) / contactFeet.length
      polyCy = contactFeet.reduce((s, fp) => s + safeNum(fp[1]), 0) / contactFeet.length
    }

    // Stability margin: min dist from ZMP to any contact foot
    let margin: number | null = null
    if (zmpAvail && contactFeet.length > 0) {
      let minD = Infinity
      contactFeet.forEach(fp => {
        const dx = zx! - safeNum(fp[0])
        const dy = zy! - safeNum(fp[1])
        const d = Math.sqrt(dx * dx + dy * dy)
        if (d < minD) minD = d
      })
      if (isFinite(minD)) margin = minD
    }

    // Escape count: frames in history where ZMP is valid but far from polygon center (>0.15m)
    let escapeCount = 0
    history.slice(-200).forEach(s => {
      const szx = validNum(s.zmp_position[0])
      const szy = validNum(s.zmp_position[1])
      if (szx === null || szy === null) return
      const pFeet = s.foot_positions
        .map((fp, i) => ({ fp, contact: s.foot_contact[i] }))
        .filter(d => d.contact)
        .map(d => d.fp)
      if (pFeet.length === 0) return
      const pCx = pFeet.reduce((sum, fp) => sum + safeNum(fp[0]), 0) / pFeet.length
      const pCy = pFeet.reduce((sum, fp) => sum + safeNum(fp[1]), 0) / pFeet.length
      if (Math.sqrt((szx - pCx) ** 2 + (szy - pCy) ** 2) > 0.15) escapeCount++
    })

    const marginLevel: StatLevel =
      margin === null ? 'warn'
      : margin > 0.05 ? 'ok'
      : margin > 0.01 ? 'warn'
      : 'err'

    return { zx, zy, zmpAvail, polyCx, polyCy, margin, marginLevel, escapeCount }
  }, [state, history])

  const NA = <span className="gait-stat-val">—</span>

  return (
    <section className="panel gait-mini-panel">
      <h2 className="panel-title">ZMP Stats</h2>
      <div className="gait-stat-rows">
        <div className="gait-stat-row">
          <span className="gait-stat-key">Current</span>
          {stats?.zmpAvail
            ? <span className="gait-stat-val">{stats.zx!.toFixed(3)}, {stats.zy!.toFixed(3)}</span>
            : NA}
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Polygon Center</span>
          {stats && stats.polyCx !== null && stats.polyCy !== null
            ? <span className="gait-stat-val">{stats.polyCx.toFixed(3)}, {stats.polyCy.toFixed(3)}</span>
            : NA}
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Stability Margin</span>
          {stats && stats.margin !== null
            ? <span className={statClass(stats.marginLevel)}>{stats.margin.toFixed(3)} m</span>
            : NA}
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Escape Count</span>
          {stats
            ? <span className={statClass(stats.escapeCount > 0 ? 'warn' : 'ok')}>{stats.escapeCount}</span>
            : NA}
        </div>
      </div>
    </section>
  )
}

// ── CoM State Panel ───────────────────────────────────────────────────────────

interface ComStatePanelProps {
  state: GaitWsPayload | null
  history: GaitWsPayload[]
}

const COM_REF_HEIGHT = 0.45

function ComStatePanel({ state, history }: ComStatePanelProps) {
  const stats = useMemo(() => {
    if (!state) return null

    const cx = safeNum(state.com_position[0])
    const cy = safeNum(state.com_position[1])
    const cz = safeNum(state.com_position[2])

    // Speed from last 2 history entries
    let speed = 0
    if (history.length >= 2) {
      const prev = history[history.length - 2]
      const curr = history[history.length - 1]
      const dt = curr.timestamp - prev.timestamp
      if (dt > 0) {
        const dx = safeNum(curr.com_position[0]) - safeNum(prev.com_position[0])
        const dy = safeNum(curr.com_position[1]) - safeNum(prev.com_position[1])
        speed = Math.sqrt(dx * dx + dy * dy) / dt
      }
    }

    // Height deviation from reference
    const heightDev = cz - COM_REF_HEIGHT
    const heightLevel: StatLevel =
      Math.abs(heightDev) < 0.03 ? 'ok' : Math.abs(heightDev) < 0.06 ? 'warn' : 'err'

    // Oscillation amplitude: max(z) - min(z) over last 50 entries
    const window50 = history.slice(-50)
    let oscAmp = 0
    if (window50.length > 1) {
      const zVals = window50.map(s => safeNum(s.com_position[2]))
      oscAmp = Math.max(...zVals) - Math.min(...zVals)
    }

    return { cx, cy, cz, speed, heightDev, heightLevel, oscAmp }
  }, [state, history])

  if (!stats) {
    return (
      <section className="panel gait-mini-panel">
        <h2 className="panel-title">CoM State</h2>
        <div className="gait-stat-rows">
          <div className="gait-stat-row"><span className="gait-stat-key">Position</span><span className="gait-stat-val">—</span></div>
          <div className="gait-stat-row"><span className="gait-stat-key">Speed</span><span className="gait-stat-val">—</span></div>
          <div className="gait-stat-row"><span className="gait-stat-key">Height Deviation</span><span className="gait-stat-val">—</span></div>
          <div className="gait-stat-row"><span className="gait-stat-key">Oscillation Amp</span><span className="gait-stat-val">—</span></div>
        </div>
      </section>
    )
  }

  return (
    <section className="panel gait-mini-panel">
      <h2 className="panel-title">CoM State</h2>
      <div className="gait-stat-rows">
        <div className="gait-stat-row">
          <span className="gait-stat-key">Position</span>
          <span className="gait-stat-val">{stats.cx.toFixed(2)}, {stats.cy.toFixed(2)}, {stats.cz.toFixed(2)}</span>
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Speed</span>
          <span className="gait-stat-val">{stats.speed.toFixed(3)} m/s</span>
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Height Deviation</span>
          <span className={statClass(stats.heightLevel)}>
            {stats.heightDev >= 0 ? '+' : ''}{stats.heightDev.toFixed(3)} m
          </span>
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Oscillation Amp</span>
          <span className="gait-stat-val">{stats.oscAmp.toFixed(4)} m</span>
        </div>
      </div>
    </section>
  )
}

// ── Head Pose Panel ───────────────────────────────────────────────────────────

interface HeadPosePanelProps {
  state: GaitWsPayload | null
  history: GaitWsPayload[]
}

function quatToRPY(w: number, qx: number, qy: number, qz: number): [number, number, number] {
  const PI = Math.PI
  const roll = Math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)) * 180 / PI
  const sinP = 2 * (w * qy - qz * qx)
  const pitch = Math.asin(Math.max(-1, Math.min(1, sinP))) * 180 / PI
  const yaw = Math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) * 180 / PI
  return [roll, pitch, yaw]
}

function HeadPosePanel({ state, history }: HeadPosePanelProps) {
  const stats = useMemo(() => {
    if (!state?.body_pose) return null

    const bp = state.body_pose
    // body_pose = [x, y, z, w, qx, qy, qz]
    const w = safeNum(bp[3])
    const qx = safeNum(bp[4])
    const qy = safeNum(bp[5])
    const qz = safeNum(bp[6])

    const [roll, pitch, yaw] = quatToRPY(w, qx, qy, qz)

    // Vibration RMS from last 50 entries
    const window50 = history.slice(-50)
    let vibRms = 0
    if (window50.length > 1) {
      const rolls: number[] = []
      const pitches: number[] = []
      window50.forEach(s => {
        if (!s.body_pose) return
        const bp2 = s.body_pose
        const [r, p] = quatToRPY(safeNum(bp2[3]), safeNum(bp2[4]), safeNum(bp2[5]), safeNum(bp2[6]))
        rolls.push(r)
        pitches.push(p)
      })
      if (rolls.length > 0) {
        const meanR = rolls.reduce((a, b) => a + b, 0) / rolls.length
        const meanP = pitches.reduce((a, b) => a + b, 0) / pitches.length
        const variance = rolls.reduce((sum, r, i) => {
          return sum + Math.pow(r - meanR, 2) + Math.pow(pitches[i] - meanP, 2)
        }, 0) / rolls.length
        vibRms = Math.sqrt(variance)
      }
    }

    const angleLevel = (deg: number): StatLevel =>
      Math.abs(deg) < 5 ? 'ok' : Math.abs(deg) < 10 ? 'warn' : 'err'
    const vibLevel: StatLevel = vibRms < 3 ? 'ok' : vibRms < 6 ? 'warn' : 'err'

    return { roll, pitch, yaw, vibRms, rollLevel: angleLevel(roll), pitchLevel: angleLevel(pitch), vibLevel }
  }, [state, history])

  if (!stats) {
    return (
      <section className="panel gait-mini-panel">
        <h2 className="panel-title">Head Pose</h2>
        <div className="gait-stat-rows">
          <div className="gait-stat-row"><span className="gait-stat-key">Roll</span><span className="gait-stat-val">—</span></div>
          <div className="gait-stat-row"><span className="gait-stat-key">Pitch</span><span className="gait-stat-val">—</span></div>
          <div className="gait-stat-row"><span className="gait-stat-key">Yaw</span><span className="gait-stat-val">—</span></div>
          <div className="gait-stat-row"><span className="gait-stat-key">Vibration RMS</span><span className="gait-stat-val">—</span></div>
        </div>
      </section>
    )
  }

  return (
    <section className="panel gait-mini-panel">
      <h2 className="panel-title">Head Pose</h2>
      <div className="gait-stat-rows">
        <div className="gait-stat-row">
          <span className="gait-stat-key">Roll</span>
          <span className={statClass(stats.rollLevel)}>{stats.roll.toFixed(2)}°</span>
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Pitch</span>
          <span className={statClass(stats.pitchLevel)}>{stats.pitch.toFixed(2)}°</span>
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Yaw</span>
          <span className="gait-stat-val">{stats.yaw.toFixed(2)}°</span>
        </div>
        <div className="gait-stat-row">
          <span className="gait-stat-key">Vibration RMS</span>
          <span className={statClass(stats.vibLevel)}>{stats.vibRms.toFixed(2)}°</span>
        </div>
      </div>
    </section>
  )
}

// ── Main Component ───────────────────────────────────────────────────────────

const DEFAULT_CONFIG: CollectionConfig = {
  num_episodes: 10,
  episode_duration: 60,
}

export default function GaitDataCollection() {
  const { state, connected, history } = useGaitWebSocket()
  const { episodes, loading: epLoading } = useGaitEpisodes(3000)
  const { startCollection, stopCollection, busy, error: ctrlError } = useCollectionControls()

  const [config, setConfig] = useState<CollectionConfig>(DEFAULT_CONFIG)

  // CoM trajectory trail — reset on new episode
  const [comTrail, setComTrail] = useState<Array<{ x: number; y: number }>>([])
  const lastEpisodeRef = useRef<number>(-1)

  // ZMP trail — rolling 50 points
  const [zmpTrail, setZmpTrail] = useState<Array<{ x: number; y: number }>>([])

  useEffect(() => {
    if (!state) return

    const epIdx = state.collection?.current_episode ?? 0
    if (epIdx !== lastEpisodeRef.current) {
      lastEpisodeRef.current = epIdx
      setComTrail([])
    }

    const [cx, cy] = state.com_position
    if (isFinite(cx) && isFinite(cy)) {
      setComTrail(prev => {
        const next = [...prev, { x: cx, y: cy }]
        return next.length > 500 ? next.slice(next.length - 500) : next
      })
    }

    const [zx, zy] = state.zmp_position
    if (isFinite(zx) && isFinite(zy)) {
      setZmpTrail(prev => {
        const next = [...prev, { x: zx, y: zy }]
        return next.length > 50 ? next.slice(next.length - 50) : next
      })
    }
  }, [state])

  const collection = state?.collection
  const isCollecting = collection?.active ?? false
  const episodeProgress = collection
    ? `${collection.current_episode} / ${collection.total_episodes}`
    : '— / —'
  const epDuration = collection?.episode_duration ?? 0

  const handleToggleCollection = useCallback(async () => {
    if (isCollecting) {
      await stopCollection()
    } else {
      await startCollection(config)
    }
  }, [isCollecting, stopCollection, startCollection, config])

  const contact = state?.foot_contact ?? [false, false, false, false]

  return (
    <div className="tab-content">
      {/* Connection warning */}
      <ConnectionBanner connected={connected} />

      {/* ── Auto-Collection Controls ─────────────────────────────────── */}
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Auto-Collection Controls</h2>
          {isCollecting && (
            <div className="gait-collecting-badge">
              <span className="rec-badge-dot" />
              Collecting
            </div>
          )}
        </div>

        <div className="gait-ctrl-row">
          <button
            className={`gait-ctrl-btn ${isCollecting ? 'gait-ctrl-btn--stop' : 'gait-ctrl-btn--start'}`}
            onClick={handleToggleCollection}
            disabled={busy}
          >
            {busy ? (
              <span className="spinner" />
            ) : isCollecting ? (
              <>
                <span className="record-icon-stop" />
                Stop
              </>
            ) : (
              <>
                <span className="record-icon-start" />
                Start Collection
              </>
            )}
          </button>

          <div className="gait-stat-pill">
            <span className="gait-stat-pill-label">Episodes</span>
            <span className="gait-stat-pill-value">{episodeProgress}</span>
          </div>

          <div className="gait-stat-pill">
            <span className="gait-stat-pill-label">Duration</span>
            <span className="gait-stat-pill-value">{formatDuration(epDuration)}</span>
          </div>
        </div>

        {collection && collection.total_episodes > 0 && (
          <div className="gait-progress-row">
            <ProgressBar
              current={collection.current_episode}
              total={collection.total_episodes}
            />
            <span className="gait-progress-label">
              {((collection.current_episode / collection.total_episodes) * 100).toFixed(0)}%
            </span>
          </div>
        )}

        <div className="gait-config-section">
          <div className="gait-config-section-title">Collection Settings</div>
          <div className="gait-config-row">
            <label className="gait-config-field">
              <span className="gait-config-label">Num Episodes</span>
              <input
                type="number"
                className="gait-config-input"
                value={config.num_episodes ?? 10}
                min={1}
                max={1000}
                disabled={isCollecting}
                onChange={e => setConfig(c => ({ ...c, num_episodes: Number(e.target.value) }))}
              />
            </label>
            <label className="gait-config-field">
              <span className="gait-config-label">Duration (sec)</span>
              <input
                type="number"
                className="gait-config-input"
                value={config.episode_duration ?? 60}
                min={5}
                max={600}
                disabled={isCollecting}
                onChange={e => setConfig(c => ({ ...c, episode_duration: Number(e.target.value) }))}
              />
            </label>
          </div>
        </div>

        {ctrlError && <div className="error-banner">{ctrlError}</div>}
      </section>

      {/* ── 2D Trajectory Monitor ────────────────────────────────────── */}
      <section className="panel gait-panel-trajectory-new">
        <div className="panel-header">
          <h2 className="panel-title">2D Trajectory Monitor</h2>
        </div>

        <div className="gait-traj-monitor-layout">
          {/* Left: canvas top-down view */}
          <div className="gait-traj-canvas-wrap">
            <CanvasTrajectory state={state} comTrail={comTrail} zmpTrail={zmpTrail} />
            {/* Legend */}
            <div className="traj-canvas-legend">
              <span className="traj-legend-item">
                <span className="traj-legend-swatch" style={{ background: '#639922', borderRadius: '50%' }} />
                CoM
              </span>
              <span className="traj-legend-item">
                <span className="traj-legend-swatch" style={{ background: '#378ADD', borderRadius: '50%' }} />
                ZMP
              </span>
              <span className="traj-legend-item">
                <span className="traj-legend-swatch" style={{ background: '#D85A30', borderRadius: '50%' }} />
                Contact
              </span>
              <span className="traj-legend-item">
                <span className="traj-legend-swatch" style={{ background: 'transparent', border: '1.5px solid #F0997B', borderRadius: '50%' }} />
                Airborne
              </span>
              <span className="traj-legend-item">
                <span
                  className="traj-legend-swatch"
                  style={{
                    background: '#7B6BAA',
                    transform: 'rotate(45deg)',
                    borderRadius: '1px',
                  }}
                />
                Waypoint
              </span>
            </div>
          </div>

          {/* Right: ZMP + CoM time series stacked */}
          <div className="gait-traj-timeseries-col">
            <div className="gait-traj-ts-panel">
              <div className="gait-traj-ts-header">
                <span className="gait-traj-ts-title">ZMP Position</span>
                <span className="gait-traj-ts-legend">
                  <span style={{ color: '#378ADD' }}>— x</span>
                  <span style={{ color: '#5DCAA5' }}>— y</span>
                </span>
              </div>
              <ZmpTimeSeries history={history} />
            </div>

            <div className="gait-traj-ts-panel">
              <div className="gait-traj-ts-header">
                <span className="gait-traj-ts-title">CoM Position</span>
                <span className="gait-traj-ts-legend">
                  <span style={{ color: '#7AABB3' }}>— x</span>
                  <span style={{ color: '#9B8EC7' }}>— y</span>
                  <span style={{ color: '#BDA6CE' }}>- - z</span>
                </span>
              </div>
              <ComTimeSeries history={history} />
            </div>

            {/* Foot contact state pills inline */}
            <div className="gait-traj-ts-panel gait-traj-ts-panel--contact">
              <div className="gait-traj-ts-header">
                <span className="gait-traj-ts-title">Foot Contact</span>
              </div>
              <div className="gait-contact-indicators" style={{ marginBottom: 0 }}>
                {FOOT_LABELS.map((label, i) => (
                  <div
                    key={label}
                    className={`gait-contact-pill ${contact[i] ? 'gait-contact-pill--on' : ''}`}
                    style={{
                      borderColor: contact[i] ? FOOT_COLORS[i] : 'rgba(180,211,217,0.3)',
                      color: contact[i] ? FOOT_COLORS[i] : '#A496B0',
                      background: contact[i] ? `${FOOT_COLORS[i]}18` : 'transparent',
                    }}
                  >
                    {label}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Navigation Trajectory (episode-level) ────────────────────── */}
      <section className="panel panel--wide">
        <div className="panel-header">
          <h2 className="panel-title">Navigation Trajectory</h2>
          <span className="gait-nav-traj-caption">Episode-level path — {comTrail.length} pts</span>
        </div>
        <NavTrajectoryChart comTrail={comTrail} state={state} />
      </section>

      {/* ── Joint Charts Row (horizontal side-by-side) ────────────────── */}
      <div className="gait-joint-horizontal">
        <section className="panel">
          <div className="panel-header">
            <h2 className="panel-title">Joint Angles</h2>
          </div>
          <JointChart history={history} showVel={false} />
        </section>
        <section className="panel">
          <div className="panel-header">
            <h2 className="panel-title">Joint Velocities</h2>
          </div>
          <JointChart history={history} showVel={true} />
        </section>
      </div>

      {/* ── Stats Row: ZMP / CoM / Head Pose ─────────────────────────── */}
      <div className="gait-stats-row">
        <ZmpStatsPanel state={state} history={history} />
        <ComStatePanel state={state} history={history} />
        <HeadPosePanel state={state} history={history} />
      </div>

      {/* Joint legend */}
      <section className="panel panel--wide">
        <h2 className="panel-title">Joint Legend</h2>
        <div className="joint-legend">
          {JOINT_NAMES.map(name => (
            <div key={name} className="legend-item">
              <svg width="20" height="8" style={{ flexShrink: 0 }}>
                <line
                  x1="0" y1="4" x2="20" y2="4"
                  stroke={jointColor(name)}
                  strokeWidth={1.5}
                  strokeDasharray={jointStrokeDash(name)}
                />
              </svg>
              <span className="legend-name">{name}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Episode Log ───────────────────────────────────────────────── */}
      <section className="panel panel--wide">
        <div className="panel-header">
          <h2 className="panel-title">Episode Log</h2>
          <span className="badge">{episodes.length} episodes</span>
        </div>

        {epLoading ? (
          <div className="loading-text">Loading episodes...</div>
        ) : episodes.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">○</div>
            <div className="empty-msg">No episodes recorded yet.</div>
            <div className="empty-sub">Start auto-collection to capture gait episodes.</div>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="episode-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Name</th>
                  <th>Duration</th>
                  <th>Steps</th>
                  <th>Goals Reached</th>
                  <th>Start Time</th>
                </tr>
              </thead>
              <tbody>
                {[...episodes].reverse().map((ep, idx) => (
                  <tr key={ep.id} className={idx === 0 ? 'row--latest' : ''}>
                    <td className="cell--id">{ep.id}</td>
                    <td className="cell--name">{ep.name}</td>
                    <td className="cell--mono">{ep.duration_sec.toFixed(2)}s</td>
                    <td className="cell--mono">{ep.n_steps.toLocaleString()}</td>
                    <td className="cell--mono">
                      {ep.waypoints_visited != null ? ep.waypoints_visited : '—'}
                    </td>
                    <td className="cell--time">{formatTime(ep.start_time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
