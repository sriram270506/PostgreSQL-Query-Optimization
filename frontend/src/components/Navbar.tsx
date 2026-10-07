import React from 'react';
import { Database, Network, Table, PlaySquare, BarChart3, Activity } from 'lucide-react';

interface NavbarProps {
  activeTab: 'tables' | 'bn' | 'cpt' | 'simulator' | 'benchmark';
  setActiveTab: (tab: 'tables' | 'bn' | 'cpt' | 'simulator' | 'benchmark') => void;
  isBackendOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, isBackendOnline }) => {
  const tabs = [
    { id: 'tables', label: 'Synthetic Data Explorer', icon: Table },
    { id: 'bn', label: 'Chow-Liu Bayesian Trees', icon: Network },
    { id: 'cpt', label: 'CPT Inspector', icon: Database },
    { id: 'simulator', label: 'FactorJoin & Query Simulator', icon: PlaySquare },
    { id: 'benchmark', label: 'Optimizer Benchmarks', icon: BarChart3 },
  ] as const;

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-indigo-600 p-0.5 shadow-lg shadow-cyan-500/20 flex items-center justify-center">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Network className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-slate-100 text-lg tracking-tight">PG Cardinality Optimizer</h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-semibold">
                  Chow-Liu + FactorJoin
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">PostgreSQL Query Selectivity & ML Fusion Dashboard</p>
            </div>
          </div>

          {/* Nav Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Backend Status Indicator */}
          <div className="flex items-center gap-2 font-mono text-xs">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${
              isBackendOnline
                ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/30'
                : 'bg-rose-950/40 text-rose-400 border-rose-500/30'
            }`}>
              <Activity className={`w-3.5 h-3.5 ${isBackendOnline ? 'animate-pulse text-emerald-400' : 'text-rose-400'}`} />
              <span>{isBackendOnline ? 'Backend: 127.0.0.1:8000' : 'Backend Offline'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Tab Switcher */}
      <div className="md:hidden flex items-center overflow-x-auto border-t border-slate-800/80 p-2 gap-1 bg-slate-950">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap ${
                isActive ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
};
