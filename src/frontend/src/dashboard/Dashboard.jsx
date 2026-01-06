import React, { useEffect, useMemo, useState, useRef } from 'react'
import { Download, CheckCircle, XCircle } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../config'
import Sidebar from './Sidebar'
import OutputPanel from './OutputPanel'
import QueueIndicator from './QueueIndicator'
import UserMenu from '../components/UserMenu'
import Footer from '../components/Footer'
import LegalModal from '../components/LegalModal'
import LoginModal from '../components/LoginModal'
import { useNSFW } from '../contexts/NSFWContext'
import { useAuth } from '../contexts/AuthContext'
import { useCredits } from '../contexts/CreditsContext'
import { TOOL_IDS } from './nav'

import TextToVideoTool from './tools/TextToVideoTool'
import ImageToVideoTool from './tools/ImageToVideoTool'
import TextToImageTool from './tools/TextToImageTool'
import TextToImageToVideoTool from './tools/TextToImageToVideoTool'
import VideoToVideoTool from './tools/VideoToVideoTool'
import VideoToTextTool from './tools/VideoToTextTool'
import SpeechToVideoTool from './tools/SpeechToVideoTool'
import VideoUpscalerTool from './tools/VideoUpscalerTool'
import FrameInterpolationTool from './tools/FrameInterpolationTool'
import PipelineTool from './tools/PipelineTool'
import LoRATrainingTool from './tools/LoRATrainingTool'
import ImageToTextTool from './tools/ImageToTextTool'
import PromptGeneratorTool from './tools/PromptGeneratorTool'
import ImageToImageTool from './tools/ImageToImageTool'
import UpscalerTool from './tools/UpscalerTool'
import AudioGenerationTool from './tools/AudioGenerationTool'
import VoiceCloningTool from './tools/VoiceCloningTool'
import LipSyncTool from './tools/LipSyncTool'
import ReframeTool from './tools/ReframeTool'
import FaceSwapTool from './tools/FaceSwapTool'
import ComingSoonTool from './tools/ComingSoonTool'
import MyMediaTool from './tools/MyMediaTool'
import Gallery from '../pages/Gallery'
import LogViewer from '../components/LogViewer'
import { sendClientLog } from '../logging'

export default function Dashboard() {
  const [activeToolId, setActiveToolId] = useState(TOOL_IDS.IMAGE_TO_VIDEO)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Credits context
  const {
    purchaseSuccess,
    purchaseCancelled,
    clearPurchaseSuccess,
    clearPurchaseCancelled
  } = useCredits()

  const [health, setHealth] = useState(null)
  const [restarting, setRestarting] = useState(false)

  const [output, setOutput] = useState(null)
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0)

  // Queue refresh token - incremented when a job is submitted
  const [queueRefreshToken, setQueueRefreshToken] = useState(0)

  // For I2V creations picker mode
  const [i2vCreationsMode, setI2vCreationsMode] = useState(false)
  const [i2vOnSelectImage, setI2vOnSelectImage] = useState(null)

  // Legal modal state
  const [legalType, setLegalType] = useState(null)

  // Ref to get current tool params for JSON export
  const toolParamsRef = useRef(null)

  const checkHealth = async () => {
    try {
      const res = await fetch(`${BACKEND_BASE}/health`)
      const data = await res.json()
      setHealth(data)
    } catch (e) {
      setHealth(null)
      if (DEBUG) console.debug('⚠️ health check failed', e)
    }
  }

  useEffect(() => {
    checkHealth()
    // Poll health every 10 seconds
    const interval = setInterval(checkHealth, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleRestartBackend = async () => {
    if (restarting) return
    if (!window.confirm('Backend herstarten? Lopende jobs worden afgebroken.')) return

    setRestarting(true)
    try {
      await fetch(`${BACKEND_BASE}/restart`, { method: 'POST' })
      // Wait for backend to come back up
      await new Promise(r => setTimeout(r, 3000))
      await checkHealth()
    } catch (e) {
      console.error('Restart failed:', e)
    } finally {
      setRestarting(false)
    }
  }

  const handleDownloadParams = () => {
    const params = toolParamsRef.current
    if (!params) {
      alert('Geen parameters beschikbaar')
      return
    }
    const blob = new Blob([JSON.stringify(params, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${activeToolId}_params_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const toolTitle = useMemo(() => {
    switch (activeToolId) {
      case TOOL_IDS.TEXT_TO_VIDEO:
        return 'Text to Video'
      case TOOL_IDS.IMAGE_TO_VIDEO:
        return 'Image to Video'
      case TOOL_IDS.TEXT_TO_IMAGE_TO_VIDEO:
        return 'Text to Image to Video'
      case TOOL_IDS.VIDEO_TO_VIDEO:
        return 'Video to Video'
      case TOOL_IDS.VIDEO_TO_TEXT:
        return 'Video to Text'
      case TOOL_IDS.VIDEO_UPSCALER:
        return 'Video Upscaler'
      case TOOL_IDS.FRAME_INTERPOLATION:
        return 'Frame Interpolation'
      case TOOL_IDS.PIPELINE:
        return 'Pipeline'
      case TOOL_IDS.LORA_TRAINING:
        return 'LoRA Training'
      case TOOL_IDS.TEXT_TO_IMAGE:
        return 'Text to Image'
      case TOOL_IDS.IMAGE_TO_IMAGE:
        return 'Image to Image'
      case TOOL_IDS.REFRAME:
        return 'Reframe'
      case TOOL_IDS.FACE_SWAP:
        return 'Face Swap'
      case TOOL_IDS.UPSCALER:
        return 'Upscaler'
      case TOOL_IDS.IMAGE_TO_TEXT:
        return 'Image to Text'
      case TOOL_IDS.PROMPT_GENERATOR:
        return 'Prompt Generator'
      case TOOL_IDS.AUDIO_GENERATION:
        return 'Audio Generation'
      case TOOL_IDS.VOICE_CLONING:
        return 'Voice Cloning'
      case TOOL_IDS.LIP_SYNC:
        return 'Lip Sync'
      case TOOL_IDS.SPEECH_TO_VIDEO:
        return 'Speech to Video'
      case TOOL_IDS.MY_MEDIA_ALL:
        return 'My Media - All'
      case TOOL_IDS.MY_MEDIA_VIDEOS:
        return 'My Media - Videos'
      case TOOL_IDS.MY_MEDIA_IMAGES:
        return 'My Media - Images'
      case TOOL_IDS.MY_MEDIA_PROMPTS:
        return 'My Media - Prompts'
      case TOOL_IDS.GALLERY:
        return 'Community Gallery'
      default:
        return 'Tool'
    }
  }, [activeToolId])

  const renderControls = () => {
    const onRefreshHistory = () => setHistoryRefreshToken((n) => n + 1)

    // Callback for I2V to enter/exit creations picker mode
    const onCreationsModeChange = (enabled, onSelect) => {
      setI2vCreationsMode(enabled)
      setI2vOnSelectImage(() => onSelect)
    }

    // Callback for tools to expose their params
    const onParamsChange = (params) => {
      toolParamsRef.current = params
    }

    // Callback for async job submission - refresh queue
    const onJobSubmitted = () => {
      setQueueRefreshToken((n) => n + 1)
    }

    switch (activeToolId) {
      case TOOL_IDS.TEXT_TO_VIDEO:
        return <TextToVideoTool onOutput={setOutput} onRefreshHistory={onRefreshHistory} onParamsChange={onParamsChange} onJobSubmitted={onJobSubmitted} />
      case TOOL_IDS.IMAGE_TO_VIDEO:
        return <ImageToVideoTool onOutput={setOutput} onRefreshHistory={onRefreshHistory} onCreationsModeChange={onCreationsModeChange} onParamsChange={onParamsChange} onJobSubmitted={onJobSubmitted} />
      case TOOL_IDS.TEXT_TO_IMAGE_TO_VIDEO:
        return <TextToImageToVideoTool onOutput={setOutput} onParamsChange={onParamsChange} onJobSubmitted={onJobSubmitted} />
      case TOOL_IDS.PIPELINE:
        return <PipelineTool />
      case TOOL_IDS.LORA_TRAINING:
        return <LoRATrainingTool onOutput={setOutput} />

      case TOOL_IDS.MY_MEDIA_ALL:
        return <MyMediaTool filter="all" />
      case TOOL_IDS.MY_MEDIA_VIDEOS:
        return <MyMediaTool filter="video" />
      case TOOL_IDS.MY_MEDIA_IMAGES:
        return <MyMediaTool filter="image" />
      case TOOL_IDS.MY_MEDIA_AUDIO:
        return <MyMediaTool filter="audio" />
      case TOOL_IDS.MY_MEDIA_PROMPTS:
        return <MyMediaTool filter="prompts" />
      case TOOL_IDS.GALLERY:
        return <Gallery />

      case TOOL_IDS.TEXT_TO_IMAGE:
        return <TextToImageTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />

      case TOOL_IDS.IMAGE_TO_TEXT:
        return <ImageToTextTool />
      case TOOL_IDS.PROMPT_GENERATOR:
        return <PromptGeneratorTool />

      case TOOL_IDS.IMAGE_TO_IMAGE:
        return <ImageToImageTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />
      case TOOL_IDS.UPSCALER:
        return <UpscalerTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />

      case TOOL_IDS.VIDEO_TO_VIDEO:
        return <VideoToVideoTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />
      case TOOL_IDS.VIDEO_TO_TEXT:
        return <VideoToTextTool />
      case TOOL_IDS.VIDEO_UPSCALER:
        return <VideoUpscalerTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />
      case TOOL_IDS.FRAME_INTERPOLATION:
        return <FrameInterpolationTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />

      case TOOL_IDS.AUDIO_GENERATION:
        return <AudioGenerationTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />

      case TOOL_IDS.VOICE_CLONING:
        return <VoiceCloningTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />

      case TOOL_IDS.LIP_SYNC:
        return <LipSyncTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />

      case TOOL_IDS.SPEECH_TO_VIDEO:
        return <SpeechToVideoTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />

      case TOOL_IDS.REFRAME:
        return <ReframeTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />
      case TOOL_IDS.FACE_SWAP:
        return <FaceSwapTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />

      default:
        return <ComingSoonTool title={toolTitle} />
    }
  }

  const { nsfwEnabled, setNsfwEnabled } = useNSFW()
  const { user, isAdult, showLoginModal, loginModalMessage, closeLoginModal } = useAuth()

  return (
    <div className="dashboard-wrapper">
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* NSFW Toggle - only for logged-in adults */}
            {isAdult && (
              <button
                className={`nsfw-toggle ${nsfwEnabled ? 'nsfw-enabled' : 'nsfw-disabled'}`}
                onClick={() => setNsfwEnabled(!nsfwEnabled)}
                title={nsfwEnabled ? 'NSFW content visible' : 'NSFW content hidden'}
              >
                {nsfwEnabled ? '🔞 NSFW' : '🛡️ SFW'}
              </button>
            )}
            {/* Queue indicator with popup */}
            <QueueIndicator
              refreshToken={queueRefreshToken}
              onJobComplete={(job) => {
                setHistoryRefreshToken((n) => n + 1)
                if (job.output_video) {
                  setOutput({
                    kind: 'video',
                    url: `${BACKEND_BASE}${job.output_video}`,
                    backendUrl: `${BACKEND_BASE}${job.output_video}`,
                  })
                }
              }}
            />
            <button
              className="icon-btn"
              onClick={handleRestartBackend}
              disabled={restarting}
              title="Herstart Backend"
              style={{ opacity: restarting ? 0.5 : 1, fontSize: '16px' }}
            >
              {restarting ? '⏳' : '🔄'}
            </button>
            <div className="status-indicator">
              <div className={`status-dot ${health?.status === 'healthy' ? 'connected' : ''}`} />
              <span>{health?.status === 'healthy' ? 'Connected' : 'Disconnected'}</span>
            </div>
            {/* User menu */}
            <UserMenu />
          </div>
        </div>

        {/* Purchase Success/Cancel Notification */}
        {purchaseSuccess && (
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1))',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: '8px',
              padding: '12px 16px',
              margin: '0 16px 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              fontSize: '0.9rem',
              color: '#10b981',
            }}
          >
            <CheckCircle size={20} />
            <span style={{ flex: 1 }}>
              <strong>Credits purchased successfully!</strong> Your balance has been updated.
            </span>
            <button
              onClick={clearPurchaseSuccess}
              style={{
                background: 'none',
                border: 'none',
                color: '#10b981',
                cursor: 'pointer',
                padding: '4px 8px',
                fontSize: '1.2rem',
              }}
            >
              ×
            </button>
          </div>
        )}

        {purchaseCancelled && (
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              padding: '12px 16px',
              margin: '0 16px 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              fontSize: '0.9rem',
              color: '#ef4444',
            }}
          >
            <XCircle size={20} />
            <span style={{ flex: 1 }}>
              Purchase cancelled. No charges were made.
            </span>
            <button
              onClick={clearPurchaseCancelled}
              style={{
                background: 'none',
                border: 'none',
                color: '#ef4444',
                cursor: 'pointer',
                padding: '4px 8px',
                fontSize: '1.2rem',
              }}
            >
              ×
            </button>
          </div>
        )}

        {/* Full-width layout for My Media tools and Gallery */}
        {(activeToolId === TOOL_IDS.MY_MEDIA_ALL ||
          activeToolId === TOOL_IDS.MY_MEDIA_VIDEOS ||
          activeToolId === TOOL_IDS.MY_MEDIA_IMAGES ||
          activeToolId === TOOL_IDS.MY_MEDIA_AUDIO ||
          activeToolId === TOOL_IDS.MY_MEDIA_PROMPTS ||
          activeToolId === TOOL_IDS.GALLERY) ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {renderControls()}
          </div>
        ) : (
          <div className="workspace">
            <section className="controls-panel">
              <div className="panel-header" style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div className="panel-title" style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Parameters</div>
                <button
                  className="icon-btn"
                  onClick={handleDownloadParams}
                  title="Download parameters als JSON"
                  style={{ padding: '4px' }}
                >
                  <Download size={16} />
                </button>
              </div>
              <div className="panel-body">{renderControls()}</div>
            </section>

            {/* Show OutputPanel only when there's active output, otherwise show MyMediaTool */}
            {output ? (
              <OutputPanel
                output={output}
                refreshToken={historyRefreshToken}
                onSelectHistoryVideo={setOutput}
                onClose={() => setOutput(null)}
              />
            ) : (
              <section className="output-panel" style={{ display: 'flex', flexDirection: 'column' }}>
                {i2vCreationsMode && (
                  <div style={{
                    padding: '12px 16px',
                    borderBottom: '1px solid var(--border-color)',
                    backgroundColor: 'var(--bg-secondary)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      Select Image for I2V
                    </span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Click an image to use it
                    </span>
                  </div>
                )}
                <div style={{ flex: 1, overflow: 'hidden' }}>
                  <MyMediaTool
                    filter="all"
                    selectionMode={i2vCreationsMode}
                    onSelectItem={i2vOnSelectImage}
                  />
                </div>
              </section>
            )}
          </div>
        )}
      </main>
      </div>
      <Footer onShowLegal={setLegalType} />
      {legalType && (
        <LegalModal type={legalType} onClose={() => setLegalType(null)} />
      )}
      {showLoginModal && (
        <LoginModal
          message={loginModalMessage}
          onClose={closeLoginModal}
        />
      )}
      <LogViewer />
    </div>
  )
}
