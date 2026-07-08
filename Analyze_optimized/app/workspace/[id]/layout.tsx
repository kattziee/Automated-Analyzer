"use client";

import Link from "next/link";
import { usePathname, useParams } from "next/navigation";
import { Sparkles, BarChart2, Eraser, Activity, LayoutDashboard, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
// We haven't created these next imports but will build a simple component for them later
// For now, let's keep it simple.

const links = [
  { href: "/overview", label: "Overview & Schema", icon: LayoutDashboard },
  { href: "/clean", label: "Clean & Impute", icon: Eraser },
  { href: "/visualize", label: "Visualization", icon: BarChart2 },
  { href: "/model", label: "Modeling & Forecasting", icon: Activity },
];

export default function WorkspaceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const params = useParams();
  const datasetId = decodeURIComponent((params.id as string) || "");

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <aside className="w-64 border-r border-border bg-card/40 backdrop-blur flex flex-col">
        <div className="h-16 flex items-center px-6 border-b border-border/50">
          <div className="flex items-center gap-2 text-foreground font-semibold">
            <Sparkles className="w-5 h-5 text-sky-400" />
            <span>Data Workspace</span>
          </div>
        </div>
        
        <div className="p-4 flex-1">
          <div className="text-xs font-medium text-muted-foreground mb-4 uppercase tracking-wider">
            Dataset
          </div>
          <div className="mb-6 px-3 py-2 bg-secondary/50 rounded-lg text-sm text-foreground truncate" title={datasetId}>
            {datasetId}
          </div>

          <div className="text-xs font-medium text-muted-foreground mb-4 uppercase tracking-wider">
            Analysis
          </div>
          <nav className="space-y-1">
            {links.map((link) => {
              const fullHref = `/workspace/${params.id}${link.href}`;
              const isActive = pathname.startsWith(fullHref);
              return (
                <Link
                  key={link.href}
                  href={fullHref}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                    isActive 
                      ? "bg-primary text-primary-foreground font-medium shadow-sm" 
                      : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                  )}
                >
                  <link.icon className="w-4 h-4" />
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>
        
        <div className="p-4 border-t border-border/50">
          <Link href="/" className="flex items-center gap-3 px-3 py-2 rounded-md text-sm text-muted-foreground hover:bg-secondary transition-colors">
            <Settings className="w-4 h-4" />
            Workspace Settings
          </Link>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {children}
      </main>
    </div>
  );
}
