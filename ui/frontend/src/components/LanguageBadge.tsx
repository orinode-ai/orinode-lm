const COLORS: Record<string, string> = {
  en: 'bg-blue-800 text-blue-200',
  ha: 'bg-green-800 text-green-200',
  yo: 'bg-purple-800 text-purple-200',
  ig: 'bg-orange-800 text-orange-200',
  pcm: 'bg-yellow-800 text-yellow-200',
}

export default function LanguageBadge({ lang }: { lang: string }) {
  const cls = COLORS[lang.toLowerCase()] ?? 'bg-gray-700 text-gray-300'
  const label = lang === 'pcm' ? 'PCM' : lang.toUpperCase()
  return (
    <span className={`inline-block text-xs px-2 py-0.5 rounded font-mono ${cls}`}>{label}</span>
  )
}
