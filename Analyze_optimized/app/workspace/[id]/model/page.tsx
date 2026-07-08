"use client";

import { useDatasetStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BrainCircuit, PlayCircle, AlertCircle, BarChart3 } from "lucide-react";
import { motion } from "framer-motion";

export default function ModelPage() {
  const { datasetId, schema } = useDatasetStore();

  return (
    <div className="flex-1 overflow-auto bg-background/50 p-6 sm:p-10 space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Machine Learning & Forecasting</h1>
          <p className="text-muted-foreground">Train predictive models and forecast trends automatically</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
          <Card className="bg-card/40 backdrop-blur border-border/50 h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BrainCircuit className="w-5 h-5 text-indigo-400" />
                Supervised Learning (AutoML)
              </CardTitle>
              <CardDescription>Select a target column to predict.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Target Variable</label>
                <select className="w-full bg-secondary/50 border border-border/50 rounded-lg p-2.5 text-sm outline-none focus:ring-2 focus:ring-indigo-500/50">
                  <option value="" disabled selected>Select a column...</option>
                  {schema.map(col => (
                    <option key={col.column} value={col.column}>{col.column}</option>
                  ))}
                </select>
              </div>

              <Button className="w-full bg-indigo-500 hover:bg-indigo-600 text-white rounded-full">
                <PlayCircle className="w-4 h-4 mr-2" />
                Train & Compare Models
              </Button>

              <div className="p-4 rounded-xl bg-card/40 border border-dashed border-border/50 text-center text-muted-foreground text-sm flex flex-col items-center justify-center gap-2">
                <AlertCircle className="w-8 h-8 opacity-20" />
                <span>Model results will appear here</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
          <Card className="bg-card/40 backdrop-blur border-border/50 h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-sky-400" />
                Time Series Forecasting
              </CardTitle>
              <CardDescription>Forecast future values based on historical trends.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Date Column</label>
                  <select className="w-full bg-secondary/50 border border-border/50 rounded-lg p-2.5 text-sm outline-none focus:ring-2 focus:ring-sky-500/50">
                    <option value="" disabled selected>Select date...</option>
                    {schema.filter(c => c.type === "datetime" || c.type === "categorical").map(col => (
                      <option key={col.column} value={col.column}>{col.column}</option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Value Column</label>
                  <select className="w-full bg-secondary/50 border border-border/50 rounded-lg p-2.5 text-sm outline-none focus:ring-2 focus:ring-sky-500/50">
                    <option value="" disabled selected>Select value...</option>
                    {schema.filter(c => c.type === "numeric").map(col => (
                      <option key={col.column} value={col.column}>{col.column}</option>
                    ))}
                  </select>
                </div>
              </div>

              <Button className="w-full bg-sky-500 hover:bg-sky-600 text-white rounded-full">
                <PlayCircle className="w-4 h-4 mr-2" />
                Run Forecast
              </Button>
              
              <div className="p-4 rounded-xl bg-card/40 border border-dashed border-border/50 text-center text-muted-foreground text-sm flex flex-col items-center justify-center gap-2">
                <AlertCircle className="w-8 h-8 opacity-20" />
                <span>Forecast charts will appear here</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
