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

export interface MapLine {
  points: { x: number; y: number }[];
}

export interface MapPolygon {
  points: { x: number; y: number }[];
}

export interface VehicleParams {
  width: number;
  length: number;
  wheelbase: number;
  front_overhang: number;
  rear_overhang: number;
}

export interface SimulationData {
  metadata: SimulationMetadata;
  vehicle_params?: VehicleParams;
  steps: TrajectoryPoint[];
  map_lines?: MapLine[];
  map_polygons?: MapPolygon[];
}

declare global {
  interface Window {
    SIMULATION_DATA: SimulationData;
  }
}
