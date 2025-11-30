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
    const duration = data.steps.length > 0 ? data.steps[data.steps.length - 1].timestamp : 0;
    set({ data, duration, currentTime: 0 });
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
