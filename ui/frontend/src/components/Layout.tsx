import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'

const PUBLIC_NAV = [
  { to: '/', label: 'Home' },
  { to: '/playground', label: 'Playground' },
  { to: '/speech', label: 'Speech' },
  { to: '/emotion', label: 'Emotion' },
  { to: '/gender', label: 'Gender' },
  { to: '/api-console', label: 'API Console' },
]

const TRAINING_NAV = [
  { to: '/training', label: 'Dashboard' },
  { to: '/training/runs', label: 'Runs' },
  { to: '/training/compare', label: 'Compare' },
  { to: '/training/evals', label: 'Evals' },
  { to: '/training/data', label: 'Data' },
]

function NavItem({ to, label, exact = false }: { to: string; label: string; exact?: boolean }) {
  const { pathname } = useLocation()
  const active = exact ? pathname === to : (to === '/' ? pathname === '/' : pathname.startsWith(to))
  return (
    <Link
      to={to}
      className={`block px-3 py-2 rounded text-sm transition-colors ${
        active
          ? 'bg-brand-700 text-white'
          : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'
      }`}
    >
      {label}
    </Link>
  )
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-48 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col overflow-y-auto">
        <div className="px-4 py-5 border-b border-gray-800">
          <span className="text-brand-500 font-bold tracking-tight">Orinode-LM</span>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {PUBLIC_NAV.map(({ to, label }) => (
            <NavItem key={to} to={to} label={label} exact={to === '/'} />
          ))}

          <div className="pt-4 pb-1">
            <p className="px-3 text-xs text-gray-600 uppercase tracking-wider">Training</p>
          </div>
          {TRAINING_NAV.map(({ to, label }) => (
            <NavItem key={to} to={to} label={label} exact={to === '/training'} />
          ))}
        </nav>

        <div className="p-3 text-xs text-gray-600 border-t border-gray-800">
          127.0.0.1:7860
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto p-6">{children}</main>
    </div>
  )
}
