"use client";

import { useDatasetStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, LineChart, PieChart, ScatterChart, Bar, Line, Pie, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useState } from "react";
import { Activity } from "lucide-react";
import { motion } from "framer-motion";

export default function VisualizePage() {
  const { datasetId, schema } = useDatasetStore();
  
  // Dummy data for visual presentation purposes
  const dummyData = [
    { name: 'Jan', value: 400, uv: 2400 },
    { name: 'Feb', value: 300, uv: 1398 },
    { name: 'Mar', value: 200, uv: 9800 },
    { name: 'Apr', value: 278, uv: 3908 },
    { name: 'May', value: 189, uv: 4800 },
    { name: 'Jun', value: 239, uv: 3800 },
    { name: 'Jul', value: 349, uv: 4300 },
  ];

  return (
    <div className="flex-1 overflow-auto bg-background/50 p-6 sm:p-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Interactive Visualization</h1>
        <p className="text-muted-foreground">Auto-generated schema-aware charts for {datasetId || "dataset"}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
          <Card className="bg-card/40 backdrop-blur border-border/50">
            <CardHeader>
              <CardTitle className="text-lg">Distribution Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dummyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                    <XAxis dataKey="name" stroke="#a1a1aa" />
                    <YAxis stroke="#a1a1aa" />
                    <Tooltip contentStyle={{ backgroundColor: '#111113', borderColor: '#27272a', borderRadius: '8px' }} />
                    <Bar dataKey="value" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
          <Card className="bg-card/40 backdrop-blur border-border/50">
            <CardHeader>
              <CardTitle className="text-lg">Time Series Trend</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dummyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                    <XAxis dataKey="name" stroke="#a1a1aa" />
                    <YAxis stroke="#a1a1aa" />
                    <Tooltip contentStyle={{ backgroundColor: '#111113', borderColor: '#27272a', borderRadius: '8px' }} />
                    <Line type="monotone" dataKey="uv" stroke="#818cf8" strokeWidth={3} dot={{ r: 4, fill: "#818cf8" }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
