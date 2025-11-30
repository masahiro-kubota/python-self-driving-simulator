import { create } from 'zustand';
import type { SimulationData, TrajectoryPoint } from '../types';

interface SimulationState {
  data: SimulationData | null;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  playbackSpeed: number;

  // Actions
  setData: (data: SimulationData) => void;
  setCurrentTime: (time: number) => void;
  togglePlay: () => void;
  setPlaybackSpeed: (speed: number) => void;

  // Helpers
  getCurrentPoint: () => TrajectoryPoint | undefined;
}

export const useSimulationStore = create<SimulationState>((set, get) => ({
  data: null,
  currentTime: 0,
  duration: 0,
  isPlaying: false,
  playbackSpeed: 1,

  setData: (data) => {
    // Flatten the nested data structure from SimulationLog to match TrajectoryPoint interface
    // The injected data has nested 'vehicle_state' and 'action' objects, but the UI expects a flat structure.
    const rawSteps = data.steps as unknown as Record<string, unknown>[];
    const flattenedSteps = rawSteps.map((step) => {
      const vehicleState = (step.vehicle_state as Record<string, unknown>) || {};
      const action = (step.action as Record<string, unknown>) || {};

      return {
        timestamp: (step.timestamp as number) ?? 0,
        x: (vehicleState.x as number) ?? (step.x as number) ?? 0,
        y: (vehicleState.y as number) ?? (step.y as number) ?? 0,
        z: (vehicleState.z as number) ?? (step.z as number) ?? 0,
        yaw: (vehicleState.yaw as number) ?? (step.yaw as number) ?? 0,
        velocity: (vehicleState.velocity as number) ?? (step.velocity as number) ?? 0,
        acceleration: (action.acceleration as number) ?? (step.acceleration as number) ?? 0,
        steering: (action.steering as number) ?? (step.steering as number) ?? 0,
      };
    });

    const duration =
      flattenedSteps.length > 0 ? flattenedSteps[flattenedSteps.length - 1].timestamp : 0;
    set({ data: { ...data, steps: flattenedSteps }, duration, currentTime: 0 });
  },

  setCurrentTime: (time) => {
    const { duration } = get();
    set({ currentTime: Math.max(0, Math.min(time, duration)) });
  },

  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),

  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  getCurrentPoint: () => {
    const { data, currentTime } = get();
    if (!data || data.steps.length === 0) return undefined;

    // Find the closest data point to currentTime
    // We want the actual data point, not an interpolated value
    let closestIndex = 0;
    let minDiff = Math.abs(data.steps[0].timestamp - currentTime);

    for (let i = 1; i < data.steps.length; i++) {
      const diff = Math.abs(data.steps[i].timestamp - currentTime);
      if (diff < minDiff) {
        minDiff = diff;
        closestIndex = i;
      } else {
        // Since timestamps are sorted, we can break early
        break;
      }
    }

    return data.steps[closestIndex];
  },
}));
