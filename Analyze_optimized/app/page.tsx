"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { UploadCloud, FileSpreadsheet, Database, TableProperties, Sparkles, ArrowRight } from "lucide-react";
import { useRouter } from "next/navigation";
import { useDatasetStore } from "@/lib/store";

export default function LandingPage() {
  const router = useRouter();
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleUpload(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await handleUpload(e.target.files[0]);
    }
  };

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Upload failed");
      }

      const data = await res.json();
      
      // Store in Zustand
      useDatasetStore.getState().setDataset({
        datasetId: data.dataset_id,
        rows: data.rows,
        columns: data.columns,
        memory: data.memory,
        qualityScore: data.quality_score,
        domain: data.domain,
        schema: data.schema,
      });

      // Navigate to the workspace overview using the dataset_id
      router.push(`/workspace/${encodeURIComponent(data.dataset_id)}/overview`);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12 relative overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-neutral-900/40 via-background to-background">
      {/* Decorative blurred blobs */}
      <div className="absolute top-1/4 left-1/4 w-[40rem] h-[40rem] bg-indigo-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-[40rem] h-[40rem] bg-sky-500/10 rounded-full blur-[120px] pointer-events-none" />

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="max-w-3xl w-full space-y-12 relative z-10"
      >
        <div className="text-center space-y-4">
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-secondary/50 border border-border text-sm text-muted-foreground mb-4"
          >
            <Sparkles className="w-4 h-4 text-sky-400" />
            <span>Automated Data Analyzer v1.0</span>
          </motion.div>
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-br from-foreground to-foreground/70">
            Intelligent Data <br/> Workspace
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            A refined, executive-ready environment for ingesting, profiling, cleaning, modeling, and explaining your data with clarity and momentum.
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`
            relative group overflow-hidden rounded-3xl border border-dashed transition-all duration-300 ease-out
            ${isDragging 
              ? "border-sky-500 bg-sky-500/5 shadow-[0_0_40px_rgba(14,165,233,0.15)] scale-[1.02]" 
              : "border-border bg-card/40 hover:bg-card/80 hover:border-muted-foreground/50 backdrop-blur-xl"
            }
          `}
        >
          <input 
            type="file" 
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" 
            onChange={handleFileInput}
            disabled={isUploading}
            accept=".csv,.tsv,.json,.xls,.xlsx,.ods,.parquet"
          />
          
          <div className="p-12 sm:p-20 flex flex-col items-center justify-center text-center space-y-6">
            <div className={`p-5 rounded-full transition-colors duration-300 ${isDragging ? 'bg-sky-500/20' : 'bg-secondary'}`}>
              <UploadCloud className={`w-10 h-10 ${isDragging ? 'text-sky-400' : 'text-muted-foreground'} transition-colors duration-300`} />
            </div>
            
            <div className="space-y-2">
              <h3 className="text-xl font-semibold text-foreground">
                {isUploading ? "Processing dataset..." : "Drag & drop to begin"}
              </h3>
              <p className="text-muted-foreground">
                or click anywhere in this area to browse
              </p>
            </div>

            <div className="flex gap-4 pt-4 justify-center text-muted-foreground/70 text-sm font-medium">
              <span className="flex items-center gap-1.5"><FileSpreadsheet className="w-4 h-4"/> CSV / Excel</span>
              <span className="flex items-center gap-1.5"><TableProperties className="w-4 h-4"/> Parquet</span>
              <span className="flex items-center gap-1.5"><Database className="w-4 h-4"/> JSON</span>
            </div>
          </div>
          
          {isUploading && (
            <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-secondary overflow-hidden">
              <motion.div 
                className="h-full bg-gradient-to-r from-sky-400 to-indigo-500"
                initial={{ width: "0%" }}
                animate={{ width: "100%" }}
                transition={{ duration: 2, ease: "easeInOut", repeat: Infinity }}
              />
            </div>
          )}
        </motion.div>

        {error && (
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            className="p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-center max-w-xl mx-auto text-sm"
          >
            {error}
          </motion.div>
        )}

      </motion.div>
    </div>
  );
}
