import { useRef } from 'react'

interface Props {
  onFile: (file: File) => void
  accept?: string
  disabled?: boolean
}

export default function AudioInput({ onFile, accept = '.wav,.flac,.mp3,.ogg', disabled }: Props) {
  const ref = useRef<HTMLInputElement>(null)

  return (
    <label className="block">
      <span className="text-sm text-gray-400 mb-1 block">Audio file (WAV / FLAC / MP3)</span>
      <input
        ref={ref}
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
        }}
        className="block w-full text-sm text-gray-400
                   file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0
                   file:text-sm file:bg-brand-700 file:text-white
                   hover:file:bg-brand-600 cursor-pointer disabled:opacity-50"
      />
    </label>
  )
}
