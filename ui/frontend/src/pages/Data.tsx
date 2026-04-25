export default function Data() {
  const langs = [
    { code: 'en', name: 'Nigerian English', corpus: 'AfriSpeech-200 + NaijaVoices' },
    { code: 'ha', name: 'Hausa', corpus: 'Common Voice + BibleTTS' },
    { code: 'ig', name: 'Igbo', corpus: 'Common Voice + BibleTTS' },
    { code: 'yo', name: 'Yoruba', corpus: 'Common Voice + BibleTTS' },
    { code: 'pcm', name: 'Nigerian Pidgin', corpus: 'Crowdsourced CS corpus' },
  ]

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-100 mb-6">Data Sources</h1>

      <p className="text-sm text-gray-400 mb-6">
        Manifests live in <code className="text-brand-400">workspace/data/manifests/</code>.
        Run <code className="text-brand-400">make build-manifests</code> to populate them.
      </p>

      <div className="space-y-3">
        {langs.map((l) => (
          <div
            key={l.code}
            className="bg-gray-900 border border-gray-800 rounded-lg p-4 flex items-center gap-4"
          >
            <span className="text-lg font-bold text-brand-400 w-10 shrink-0">{l.code.toUpperCase()}</span>
            <div>
              <div className="text-sm font-semibold text-gray-200">{l.name}</div>
              <div className="text-xs text-gray-500">{l.corpus}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8 p-4 bg-gray-900 border border-gray-800 rounded-lg text-xs text-gray-500">
        <p className="font-semibold text-gray-400 mb-2">Quick start</p>
        <pre className="text-gray-400 whitespace-pre-wrap">{`make download-data    # download all corpora
make build-manifests  # resample → 16kHz FLAC + JSONL
make validate-data    # diacritic sanity check`}</pre>
      </div>
    </div>
  )
}
