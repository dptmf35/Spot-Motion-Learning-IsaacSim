import { useState, useEffect, useCallback } from 'react'

const GAIT_BASE = 'http://localhost:8001'

export interface GaitApiStatus {
  collection_active: boolean
  current_episode: number
  total_episodes: number
  episode_duration: number
  recording: boolean
  episode_count: number
  recording_duration: number
  connected_to_sim: boolean
}

export interface GaitEpisode {
  id: string
  name: string
  n_steps: number
  duration_sec: number
  start_time: string
  waypoints_visited?: number
}

export interface CollectionConfig {
  num_episodes?: number
  episode_duration?: number
}

export interface StartCollectionResponse {
  status: string
  config: CollectionConfig
}

export interface StopCollectionResponse {
  status: string
}

async function gaitFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${GAIT_BASE}${path}`, options)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export function useGaitStatus(intervalMs = 2000) {
  const [status, setStatus] = useState<GaitApiStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true

    const poll = async () => {
      try {
        const data = await gaitFetch<GaitApiStatus>('/api/status')
        if (alive) {
          setStatus(data)
          setError(null)
        }
      } catch (e) {
        if (alive) setError(String(e))
      }
    }

    poll()
    const id = setInterval(poll, intervalMs)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [intervalMs])

  return { status, error }
}

export function useCollectionControls() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startCollection = useCallback(async (config: CollectionConfig): Promise<StartCollectionResponse | null> => {
    setBusy(true)
    setError(null)
    try {
      const data = await gaitFetch<StartCollectionResponse>('/api/collection/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      return data
    } catch (e) {
      setError(String(e))
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  const stopCollection = useCallback(async (): Promise<StopCollectionResponse | null> => {
    setBusy(true)
    setError(null)
    try {
      const data = await gaitFetch<StopCollectionResponse>('/api/collection/stop', { method: 'POST' })
      return data
    } catch (e) {
      setError(String(e))
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  const updateConfig = useCallback(async (config: CollectionConfig): Promise<void> => {
    setBusy(true)
    setError(null)
    try {
      await gaitFetch('/api/collection/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }, [])

  return { startCollection, stopCollection, updateConfig, busy, error }
}

export function useGaitEpisodes(intervalMs = 3000) {
  const [episodes, setEpisodes] = useState<GaitEpisode[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true

    const poll = async () => {
      try {
        const data = await gaitFetch<GaitEpisode[]>('/api/episodes')
        if (alive) {
          setEpisodes(data)
          setLoading(false)
        }
      } catch {
        if (alive) setLoading(false)
      }
    }

    poll()
    const id = setInterval(poll, intervalMs)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [intervalMs])

  return { episodes, loading }
}
