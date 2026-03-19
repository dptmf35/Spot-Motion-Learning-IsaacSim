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
  BarChart,
  Bar,
  Cell,
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

// Each leg has a base color; stroke style varies by joint type (hip_x/hip_y/knee)
const LEG_COLORS: Record<string, string> = {
  FL: '#7AABB3',  // dark teal
  FR: '#9B8EC7',  // purple
  RL: '#BDA6CE',  // lavender
  RR: '#6B5F7A',  // muted purple-grey
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
  return '0'  // solid
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

interface TrajectoryChartProps {
  state: GaitWsPayload | null
  comTrail: Array<{ x: number; y: number }>
  zmpTrail: Array<{ x: number; y: number }>
}

function TrajectoryChart({ state, comTrail, zmpTrail }: TrajectoryChartProps) {
  const footData = useMemo(() => {
    if (!state) return []
    return state.foot_positions.map((fp, i) => ({
      x: safeNum(fp[0]),
      y: safeNum(fp[1]),
      label: FOOT_LABELS[i],
      color: FOOT_COLORS[i],
      contact: state.foot_contact[i],
    }))
  }, [state])

  const waypoint = state?.collection?.current_waypoint ?? null
  const headPt = state ? [{ x: safeNum(state.head_position[0]), y: safeNum(state.head_position[1]) }] : []
  const wayptData = waypoint ? [{ x: safeNum(waypoint[0]), y: safeNum(waypoint[1]) }] : []

  // Compute axis domain with some padding
  const allX = [
    ...comTrail.map(p => p.x),
    ...footData.map(p => p.x),
    ...(waypoint ? [waypoint[0]] : []),
  ].filter(isFinite)
  const allY = [
    ...comTrail.map(p => p.y),
    ...footData.map(p => p.y),
    ...(waypoint ? [waypoint[1]] : []),
  ].filter(isFinite)

  const pad = 0.5
  const xMin = allX.length ? Math.min(...allX) - pad : -2
  const xMax = allX.length ? Math.max(...allX) + pad : 2
  const yMin = allY.length ? Math.min(...allY) - pad : -2
  const yMax = allY.length ? Math.max(...allY) + pad : 2

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(180,211,217,0.3)" />
        <XAxis
          dataKey="x"
          type="number"
          domain={[xMin, xMax]}
          name="X"
          unit="m"
          tick={{ fill: '#6B5F7A', fontSize: 10 }}
          stroke="rgba(180,211,217,0.4)"
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <YAxis
          dataKey="y"
          type="number"
          domain={[yMin, yMax]}
          name="Y"
          unit="m"
          width={40}
          tick={{ fill: '#6B5F7A', fontSize: 10 }}
          stroke="rgba(180,211,217,0.4)"
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <Tooltip
          {...TOOLTIP_STYLE}
          cursor={false}
          formatter={(v: number) => v.toFixed(3)}
        />

        {/* CoM trail — teal dots connected visually by density */}
        <Scatter
          name="CoM Path"
          data={comTrail}
          fill="rgba(180,211,217,0.6)"
          line={{ stroke: '#B4D3D9', strokeWidth: 1.5, opacity: 0.7 }}
          lineType="joint"
          shape={<circle r={1.5} />}
        />

        {/* ZMP trail — lavender dots */}
        <Scatter
          name="ZMP"
          data={zmpTrail}
          fill="rgba(189,166,206,0.7)"
          shape={<circle r={2} />}
        />

        {/* Head position — purple */}
        <Scatter
          name="Head"
          data={headPt}
          fill="#9B8EC7"
          shape={<circle r={5} />}
        />

        {/* Current waypoint — dark purple diamond marker */}
        <Scatter
          name="Waypoint"
          data={wayptData}
          fill="#7B6BAA"
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

        {/* Foot positions — colored by leg, ring if airborne */}
        {footData.map(fp => (
          <Scatter
            key={fp.label}
            name={fp.label}
            data={[{ x: fp.x, y: fp.y }]}
            fill={fp.contact ? fp.color : 'transparent'}
            stroke={fp.color}
            strokeWidth={fp.contact ? 0 : 2}
            shape={<circle r={5} />}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
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

interface ContactBarsProps {
  contact: boolean[]
}

function ContactBars({ contact }: ContactBarsProps) {
  const data = FOOT_LABELS.map((label, i) => ({
    label,
    value: contact[i] ? 1 : 0,
    color: contact[i] ? FOOT_COLORS[i] : 'rgba(180,211,217,0.2)',
  }))

  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(180,211,217,0.3)" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: '#6B5F7A', fontSize: 11 }}
          stroke="rgba(180,211,217,0.4)"
        />
        <YAxis
          domain={[0, 1]}
          hide
        />
        <Bar dataKey="value" radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {data.map((entry, idx) => (
            <Cell key={idx} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

interface AirTimeBarsProps {
  history: GaitWsPayload[]
}

function AirTimeBars({ history }: AirTimeBarsProps) {
  // Compute fraction of time each foot was airborne over the rolling window
  const data = useMemo(() => {
    if (history.length === 0) {
      return FOOT_LABELS.map((label, i) => ({ label, airFraction: 0, color: FOOT_COLORS[i] }))
    }
    const counts = [0, 0, 0, 0]
    history.forEach(s => {
      s.foot_contact.forEach((c, i) => {
        if (!c) counts[i]++
      })
    })
    const n = history.length
    return FOOT_LABELS.map((label, i) => ({
      label,
      airFraction: counts[i] / n,
      color: FOOT_COLORS[i],
    }))
  }, [history])

  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(180,211,217,0.3)" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: '#6B5F7A', fontSize: 11 }}
          stroke="rgba(180,211,217,0.4)"
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          tick={{ fill: '#6B5F7A', fontSize: 10 }}
          stroke="rgba(180,211,217,0.4)"
          width={36}
        />
        <Tooltip
          {...TOOLTIP_STYLE}
          formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
        />
        <Bar dataKey="airFraction" radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {data.map((entry, idx) => (
            <Cell key={idx} fill={entry.color} fillOpacity={0.7} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
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
  const [showVel, setShowVel] = useState(false)

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
          {/* Start / Stop */}
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

          {/* Episode progress */}
          <div className="gait-stat-pill">
            <span className="gait-stat-pill-label">Episodes</span>
            <span className="gait-stat-pill-value">{episodeProgress}</span>
          </div>

          {/* Episode duration countdown */}
          <div className="gait-stat-pill">
            <span className="gait-stat-pill-label">Duration</span>
            <span className="gait-stat-pill-value">{formatDuration(epDuration)}</span>
          </div>
        </div>

        {/* Progress bar */}
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

        {/* Config inputs */}
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

      {/* ── Charts Row 1: Trajectory + Joint Angles ──────────────────── */}
      <div className="gait-charts-row">
        {/* 2D Trajectory */}
        <section className="panel gait-panel-trajectory">
          <div className="panel-header">
            <h2 className="panel-title">2D Trajectory (XY)</h2>
            <div className="gait-traj-legend">
              <span className="gait-traj-legend-item" style={{ color: '#B4D3D9' }}>— CoM</span>
              <span className="gait-traj-legend-item" style={{ color: '#BDA6CE' }}>· ZMP</span>
              <span className="gait-traj-legend-item" style={{ color: '#9B8EC7' }}>● Head</span>
              <span className="gait-traj-legend-item" style={{ color: '#7B6BAA' }}>◆ WP</span>
            </div>
          </div>
          <TrajectoryChart state={state} comTrail={comTrail} zmpTrail={zmpTrail} />
          {/* Foot position legend */}
          <div className="gait-foot-legend">
            {FOOT_LABELS.map((label, i) => (
              <span key={label} className="gait-foot-legend-item">
                <span
                  className="gait-foot-dot"
                  style={{
                    background: contact[i] ? FOOT_COLORS[i] : 'transparent',
                    border: `2px solid ${FOOT_COLORS[i]}`,
                  }}
                />
                {label}
              </span>
            ))}
          </div>
        </section>

        {/* Joint charts stacked */}
        <div className="gait-joint-col">
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">Joint Angles</h2>
              <div className="toggle-group">
                <button
                  className={`toggle-btn ${!showVel ? 'active' : ''}`}
                  onClick={() => setShowVel(false)}
                >
                  Pos (rad)
                </button>
                <button
                  className={`toggle-btn ${showVel ? 'active' : ''}`}
                  onClick={() => setShowVel(true)}
                >
                  Vel (rad/s)
                </button>
              </div>
            </div>
            <JointChart history={history} showVel={false} />
          </section>

          <section className="panel">
            <h2 className="panel-title">Joint Velocities</h2>
            <JointChart history={history} showVel={true} />
          </section>
        </div>
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

      {/* ── Charts Row 2: Contact + Air Time ─────────────────────────── */}
      <div className="gait-charts-row gait-charts-row--half">
        <section className="panel">
          <h2 className="panel-title">Foot Contact State</h2>
          <div className="gait-contact-indicators">
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
          <ContactBars contact={contact} />
        </section>

        <section className="panel">
          <h2 className="panel-title">Air Time Fraction (200-frame window)</h2>
          <AirTimeBars history={history} />
        </section>
      </div>

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
