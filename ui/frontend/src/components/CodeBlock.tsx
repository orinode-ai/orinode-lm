import { useState } from 'react'

interface Props {
  code: string
  language?: string
}

export default function CodeBlock({ code }: Props) {
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <div className="relative">
      <button
        onClick={copy}
        className="absolute top-2 right-2 text-xs px-2 py-1 rounded bg-gray-700
                   hover:bg-gray-600 text-gray-300 transition-colors"
      >
        {copied ? 'Copied!' : 'Copy'}
      </button>
      <pre className="bg-gray-950 border border-gray-800 rounded p-4 text-sm text-gray-300
                      overflow-x-auto font-mono whitespace-pre-wrap">
        {code}
      </pre>
    </div>
  )
}
