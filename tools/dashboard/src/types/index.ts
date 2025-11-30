export interface TrajectoryPoint {
  timestamp: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  velocity: number;
  acceleration: number;
  steering: number;
}

export interface SimulationMetadata {
  controller: string;
  simulator: string;
  execution_time: string;
  [key: string]: string | number;
}

export interface SimulationData {
  metadata: SimulationMetadata;
  steps: TrajectoryPoint[];
}

declare global {
  interface Window {
    SIMULATION_DATA: SimulationData;
  }
}
