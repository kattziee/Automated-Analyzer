import { create } from 'zustand';

export interface ColumnSchema {
  column: string;
  type: string;
  dtype: string;
  unique: number;
  missing_pct: number;
}

export interface DatasetState {
  datasetId: string | null;
  rows: number;
  columns: number;
  memory: string;
  qualityScore: number;
  domain: string;
  schema: ColumnSchema[];
  setDataset: (data: Partial<DatasetState>) => void;
}

export const useDatasetStore = create<DatasetState>((set) => ({
  datasetId: null,
  rows: 0,
  columns: 0,
  memory: "0 B",
  qualityScore: 0,
  domain: "Unknown",
  schema: [],
  setDataset: (data) => set((state) => ({ ...state, ...data })),
}));
