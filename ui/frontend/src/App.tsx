import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import ApiConsole from './pages/ApiConsole'
import Compare from './pages/Compare'
import Dashboard from './pages/Dashboard'
import Data from './pages/Data'
import Emotion from './pages/Emotion'
import Evals from './pages/Evals'
import Gender from './pages/Gender'
import Home from './pages/Home'
import Playground from './pages/Playground'
import RunDetail from './pages/RunDetail'
import Runs from './pages/Runs'
import SpeechRecognition from './pages/SpeechRecognition'

export default function App() {
  return (
    <Layout>
      <Routes>
        {/* Public pages */}
        <Route path="/" element={<Home />} />
        <Route path="/playground" element={<Playground />} />
        <Route path="/speech" element={<SpeechRecognition />} />
        <Route path="/emotion" element={<Emotion />} />
        <Route path="/gender" element={<Gender />} />
        <Route path="/api-console" element={<ApiConsole />} />

        {/* Training dashboard */}
        <Route path="/training" element={<Dashboard />} />
        <Route path="/training/runs" element={<Runs />} />
        <Route path="/training/runs/:runId" element={<RunDetail />} />
        <Route path="/training/compare" element={<Compare />} />
        <Route path="/training/evals" element={<Evals />} />
        <Route path="/training/data" element={<Data />} />
      </Routes>
    </Layout>
  )
}
