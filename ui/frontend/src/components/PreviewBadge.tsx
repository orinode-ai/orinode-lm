export default function PreviewBadge({ title = 'Preview' }: { title?: string }) {
  return (
    <span className="inline-block text-xs px-2 py-0.5 rounded bg-yellow-900 text-yellow-300 border border-yellow-700 font-semibold tracking-wide">
      {title}
    </span>
  )
}
