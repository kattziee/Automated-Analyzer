"use client";

import { useDatasetStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles, Database, FileText, BarChart, Layers } from "lucide-react";
import { motion } from "framer-motion";

export default function OverviewPage() {
  const { datasetId, rows, columns, memory, qualityScore, domain, schema } = useDatasetStore();

  if (!datasetId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-muted-foreground text-center">
          <Database className="w-12 h-12 mx-auto mb-4 opacity-20" />
          <p>No dataset loaded.</p>
        </div>
      </div>
    );
  }

  const kpis = [
    { label: "Rows", value: rows.toLocaleString(), icon: FileText },
    { label: "Columns", value: columns.toLocaleString(), icon: Layers },
    { label: "Memory", value: memory, icon: Database },
    { label: "Quality Score", value: `${qualityScore}/100`, icon: Sparkles },
    { label: "Domain", value: domain, icon: BarChart },
  ];

  return (
    <div className="flex-1 overflow-auto bg-background/50 p-6 sm:p-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Workspace Overview</h1>
        <p className="text-muted-foreground">High-level schema and quality profiling for {datasetId}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {kpis.map((kpi, idx) => (
          <motion.div 
            key={kpi.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
          >
            <Card className="bg-card/50 backdrop-blur border-border/50 shadow-sm hover:shadow-md transition-shadow">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  {kpi.label}
                </CardTitle>
                <kpi.icon className="w-4 h-4 text-sky-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{kpi.value}</div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold tracking-tight">Schema Inference</h2>
            <div className="h-px flex-1 bg-border/50 ml-4" />
          </div>
          
          <div className="rounded-xl border border-border/50 bg-card/40 backdrop-blur overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-secondary/50 text-muted-foreground text-xs uppercase tracking-wider border-b border-border/50">
                  <tr>
                    <th className="px-6 py-4 font-medium">Column Name</th>
                    <th className="px-6 py-4 font-medium">Type</th>
                    <th className="px-6 py-4 font-medium">DType</th>
                    <th className="px-6 py-4 font-medium">Unique</th>
                    <th className="px-6 py-4 font-medium">Missing</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30">
                  {schema.map((col, idx) => (
                    <motion.tr 
                      key={col.column}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.2 + (idx * 0.02) }}
                      className="hover:bg-secondary/20 transition-colors"
                    >
                      <td className="px-6 py-3 font-medium text-foreground">{col.column}</td>
                      <td className="px-6 py-3">
                        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20">
                          {col.type}
                        </span>
                      </td>
                      <td className="px-6 py-3 text-muted-foreground">{col.dtype}</td>
                      <td className="px-6 py-3">{col.unique.toLocaleString()}</td>
                      <td className="px-6 py-3">
                        <div className="flex items-center gap-2">
                          <span className="w-8">{col.missing_pct.toFixed(1)}%</span>
                          <div className="w-16 h-1.5 bg-secondary rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-destructive/60" 
                              style={{ width: `${col.missing_pct}%` }} 
                            />
                          </div>
                        </div>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold tracking-tight">Quality Breakdown</h2>
            <div className="h-px flex-1 bg-border/50 ml-4" />
          </div>
          <Card className="bg-card/40 backdrop-blur border-border/50 shadow-sm">
            <CardContent className="p-6 space-y-6">
              {[
                { label: "Completeness", score: 95 }, // Mock for now
                { label: "Uniqueness", score: 88 },
                { label: "Consistency", score: 100 },
                { label: "Validity", score: 92 },
              ].map((q) => (
                <div key={q.label} className="space-y-2">
                  <div className="flex justify-between text-sm font-medium">
                    <span>{q.label}</span>
                    <span className="text-muted-foreground">{q.score}/100</span>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${q.score}%` }}
                      transition={{ duration: 1, delay: 0.3 }}
                      className="h-full bg-gradient-to-r from-sky-400 to-indigo-500 rounded-full" 
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
