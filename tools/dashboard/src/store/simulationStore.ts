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
    const flattenedSteps = data.steps.map((step: any) => {
      const vehicleState = step.vehicle_state || {};
      const action = step.action || {};

      return {
        timestamp: step.timestamp,
        x: vehicleState.x ?? step.x ?? 0,
        y: vehicleState.y ?? step.y ?? 0,
        z: vehicleState.z ?? step.z ?? 0,
        yaw: vehicleState.yaw ?? step.yaw ?? 0,
        velocity: vehicleState.velocity ?? step.velocity ?? 0,
        acceleration: action.acceleration ?? step.acceleration ?? 0,
        steering: action.steering ?? step.steering ?? 0,
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
    if (!data) return undefined;

    // Simple linear search for now (can be optimized with binary search)
    // Find point with timestamp <= currentTime
    const index = data.steps.findIndex((step) => step.timestamp > currentTime);
    if (index === -1) return data.steps[data.steps.length - 1];
    if (index === 0) return data.steps[0];

    // Interpolation could be added here
    return data.steps[index - 1];
  },
}));
