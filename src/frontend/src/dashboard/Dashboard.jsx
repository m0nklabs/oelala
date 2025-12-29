import React, { useEffect, useMemo, useState } from 'react'
import { BACKEND_BASE, DEBUG } from '../config'
import Sidebar from './Sidebar'
import OutputPanel from './OutputPanel'
import { TOOL_IDS } from './nav'

import TextToVideoTool from './tools/TextToVideoTool'
import ImageToVideoTool from './tools/ImageToVideoTool'
import TextToImageTool from './tools/TextToImageTool'
import TextToImageToVideoTool from './tools/TextToImageToVideoTool'
import PipelineTool from './tools/PipelineTool'
import LoRATrainingTool from './tools/LoRATrainingTool'
import ComingSoonTool from './tools/ComingSoonTool'
import LogViewer from '../components/LogViewer'
import { sendClientLog } from '../logging'

export default function Dashboard() {
  const [activeToolId, setActiveToolId] = useState(TOOL_IDS.TEXT_TO_VIDEO)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const [health, setHealth] = useState(null)

  const [output, setOutput] = useState(null)
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0)

  useEffect(() => {
    ;(async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/health`)
        const data = await res.json()
        setHealth(data)
      } catch (e) {
        if (DEBUG) console.debug('⚠️ health check failed', e)
        await sendClientLog({
          level: 'error',
          message: 'Health check failed',
          timestamp: new Date().toISOString(),
          meta: { message: e?.message },
        })
      }
    })()
  }, [])

  const toolTitle = useMemo(() => {
    switch (activeToolId) {
      case TOOL_IDS.TEXT_TO_VIDEO:
        return 'Text to Video'
      case TOOL_IDS.IMAGE_TO_VIDEO:
        return 'Image to Video'
      case TOOL_IDS.TEXT_TO_IMAGE_TO_VIDEO:
        return 'Text to Image to Video'
      case TOOL_IDS.PIPELINE:
        return 'Pipeline'
      case TOOL_IDS.LORA_TRAINING:
        return 'LoRA Training'
      case TOOL_IDS.TEXT_TO_IMAGE:
        return 'Text to Image'
      case TOOL_IDS.VIDEO_TO_VIDEO:
        return 'Video to Video'
      case TOOL_IDS.IMAGE_TO_IMAGE:
        return 'Image to Image'
      case TOOL_IDS.REFRAME:
        return 'Reframe'
      case TOOL_IDS.FACE_SWAP:
        return 'Face Swap'
      case TOOL_IDS.UPSCALER:
        return 'Upscaler'
      default:
        return 'Tool'
    }
  }, [activeToolId])

  const renderControls = () => {
    const onRefreshHistory = () => setHistoryRefreshToken((n) => n + 1)

    switch (activeToolId) {
      case TOOL_IDS.TEXT_TO_VIDEO:
        return <TextToVideoTool onOutput={setOutput} onRefreshHistory={onRefreshHistory} />
      case TOOL_IDS.IMAGE_TO_VIDEO:
        return <ImageToVideoTool onOutput={setOutput} onRefreshHistory={onRefreshHistory} />
      case TOOL_IDS.TEXT_TO_IMAGE_TO_VIDEO:
        return <TextToImageToVideoTool onOutput={setOutput} />
      case TOOL_IDS.PIPELINE:
        return <PipelineTool />
      case TOOL_IDS.LORA_TRAINING:
        return <LoRATrainingTool onOutput={setOutput} />

      case TOOL_IDS.TEXT_TO_IMAGE:
        return <TextToImageTool onOutput={setOutput} />

      case TOOL_IDS.VIDEO_TO_VIDEO:
      case TOOL_IDS.IMAGE_TO_IMAGE:
      case TOOL_IDS.REFRAME:
      case TOOL_IDS.FACE_SWAP:
      case TOOL_IDS.UPSCALER:
        return <ComingSoonTool title={toolTitle} />

      default:
        return <ComingSoonTool title={toolTitle} />
    }
  }

  return (
    <div className="dashboard-container">
      <Sidebar
        activeToolId={activeToolId}
        onSelectTool={setActiveToolId}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
      />

      <main className="main-content">
        <div className="top-bar">
          <h1>{toolTitle}</h1>
          <div className="status-indicator">
            <div className={`status-dot ${health?.status === 'healthy' ? 'connected' : ''}`} />
            <span>{health?.status === 'healthy' ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        <div className="workspace">
          <section className="controls-panel">
            <div className="panel-header" style={{ marginBottom: '16px' }}>
              <div className="panel-title" style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Parameters</div>
            </div>
            <div className="panel-body">{renderControls()}</div>
          </section>

          <OutputPanel
            output={output}
            refreshToken={historyRefreshToken}
            onSelectHistoryVideo={setOutput}
          />
        </div>
      </main>
      <LogViewer />
    </div>
  )
}
