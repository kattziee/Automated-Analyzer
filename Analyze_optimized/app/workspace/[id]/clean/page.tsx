"use client";

import { useDatasetStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Eraser, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";
import { useState } from "react";
import { motion } from "framer-motion";

export default function CleanPage() {
  const { datasetId } = useDatasetStore();
  const [isCleaning, setIsCleaning] = useState(false);
  const [cleanStats, setCleanStats] = useState<any>(null);

  const handleClean = async () => {
    setIsCleaning(true);
    try {
      const res = await fetch(`/api/dataset/${datasetId}/clean`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setCleanStats(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsCleaning(false);
    }
  };

  return (
    <div className="flex-1 overflow-auto bg-background/50 p-6 sm:p-10 space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Data Cleaning Pipeline</h1>
          <p className="text-muted-foreground">Automated imputation, deduplication, and standardizing for {datasetId}</p>
        </div>
        <Button 
          onClick={handleClean} 
          disabled={isCleaning || !datasetId}
          className="bg-sky-500 hover:bg-sky-600 text-white rounded-full px-6 shadow-[0_0_20px_rgba(14,165,233,0.3)] transition-all"
        >
          {isCleaning ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Eraser className="w-4 h-4 mr-2" />}
          Run Cleaning Engine
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="lg:col-span-1 space-y-6">
          <Card className="bg-card/40 backdrop-blur border-border/50">
            <CardHeader>
              <CardTitle>Cleaning Parameters</CardTitle>
              <CardDescription>Configure the automated pipeline steps.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Remove Duplicates</span>
                <CheckCircle2 className="w-5 h-5 text-sky-400" />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Impute Missing Values</span>
                <CheckCircle2 className="w-5 h-5 text-sky-400" />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Drop Sparse Columns</span>
                <CheckCircle2 className="w-5 h-5 text-sky-400" />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Standardize Text</span>
                <CheckCircle2 className="w-5 h-5 text-sky-400" />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <div className="lg:col-span-2">
          {cleanStats ? (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
              <Card className="bg-sky-500/5 border-sky-500/20 backdrop-blur">
                <CardHeader>
                  <CardTitle className="text-sky-400 flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5" /> Cleaning Successful
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="p-4 rounded-xl bg-card/60 border border-border/50">
                      <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Rows Reduced</div>
                      <div className="text-xl font-bold">{cleanStats.rows_before - cleanStats.rows_after}</div>
                    </div>
                    <div className="p-4 rounded-xl bg-card/60 border border-border/50">
                      <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Cols Dropped</div>
                      <div className="text-xl font-bold">{cleanStats.dropped_columns?.length || 0}</div>
                    </div>
                    <div className="p-4 rounded-xl bg-card/60 border border-border/50">
                      <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Duplicates</div>
                      <div className="text-xl font-bold">{cleanStats.duplicates_removed}</div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold">Audit Log</h4>
                    <div className="p-4 rounded-xl bg-black/40 border border-border/50 text-sm font-mono text-muted-foreground space-y-1">
                      {cleanStats.audit_log?.map((log: string, i: number) => (
                        <div key={i} className="flex gap-2">
                          <span className="text-sky-400">{">"}</span> {log}
                        </div>
                      )) || <div>No changes made.</div>}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            <Card className="h-full bg-card/10 border-dashed border-border/60 flex flex-col items-center justify-center p-12 text-center text-muted-foreground">
              <AlertCircle className="w-12 h-12 mb-4 opacity-20" />
              <p>Run the cleaning engine to view the audit log and results.</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
