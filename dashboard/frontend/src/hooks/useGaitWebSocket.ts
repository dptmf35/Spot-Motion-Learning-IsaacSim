import { useState, useEffect, useRef, useCallback } from 'react'

export interface GaitCollectionState {
  active: boolean
  current_episode: number
  total_episodes: number
  episode_duration: number
  current_waypoint: number[] | null  // [x, y, yaw]
  waypoints_visited: number
  waypoints_per_episode: number
}

export interface GaitWsPayload {
  timestamp: number
  body_pose: number[]       // [x, y, z, w, qx, qy, qz]
  commands: number[]        // [vx, vy, wz]
  com_position: number[]    // [x, y, z]
  zmp_position: number[]    // [x, y] (may contain NaN)
  foot_positions: number[][] // [[x,y,z], [x,y,z], [x,y,z], [x,y,z]] FL/FR/RL/RR
  foot_contact: boolean[]   // [FL, FR, RL, RR]
  head_position: number[]   // [x, y, z]
  joint_pos: number[]       // 12 values
  joint_vel: number[]       // 12 values
  recording: boolean
  episode_count: number
  recording_duration: number
  collection: GaitCollectionState
}

const HISTORY_SIZE = 200
const RECONNECT_DELAY = 2000

function getGaitWsUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/ws`
}

export function useGaitWebSocket() {
  const [state, setState] = useState<GaitWsPayload | null>(null)
  const [connected, setConnected] = useState(false)
  const [history, setHistory] = useState<GaitWsPayload[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(getGaitWsUrl())
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) return
        setConnected(true)
      }

      ws.onmessage = (event: MessageEvent) => {
        if (!mountedRef.current) return
        try {
          const payload: GaitWsPayload = JSON.parse(event.data as string)
          setState(payload)
          setHistory(prev => {
            const next = [...prev, payload]
            if (next.length > HISTORY_SIZE) {
              return next.slice(next.length - HISTORY_SIZE)
            }
            return next
          })
        } catch {
          // ignore malformed frames
        }
      }

      ws.onerror = () => {
        if (!mountedRef.current) return
        setConnected(false)
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setConnected(false)
        wsRef.current = null
        reconnectTimer.current = setTimeout(() => {
          if (mountedRef.current) connect()
        }, RECONNECT_DELAY)
      }
    } catch {
      setConnected(false)
      reconnectTimer.current = setTimeout(() => {
        if (mountedRef.current) connect()
      }, RECONNECT_DELAY)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  return { state, connected, history }
}
