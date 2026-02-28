import React, { useEffect, useMemo, useState, useRef, lazy, Suspense } from 'react'
import { Download, CheckCircle, XCircle, Settings2, ChevronUp, Menu, X, Loader2, PanelRightClose, PanelRight } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../config'
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

// Lazy load all tool components for code splitting
const TextToVideoTool = lazy(() => import('./tools/TextToVideoTool'))
const ImageToVideoTool = lazy(() => import('./tools/ImageToVideoTool'))
const TextToImageTool = lazy(() => import('./tools/TextToImageTool'))
const TextToImageToVideoTool = lazy(() => import('./tools/TextToImageToVideoTool'))
const VideoToVideoTool = lazy(() => import('./tools/VideoToVideoTool'))
const VideoToTextTool = lazy(() => import('./tools/VideoToTextTool'))
const SpeechToVideoTool = lazy(() => import('./tools/SpeechToVideoTool'))
const PostProcessingTool = lazy(() => import('./tools/PostProcessingTool'))
const PipelineTool = lazy(() => import('./tools/PipelineTool'))
const LoRATrainingTool = lazy(() => import('./tools/LoRATrainingTool'))
const ImageToTextTool = lazy(() => import('./tools/ImageToTextTool'))
const PromptGeneratorTool = lazy(() => import('./tools/PromptGeneratorTool'))
const ImageToImageTool = lazy(() => import('./tools/ImageToImageTool'))
const AudioGenerationTool = lazy(() => import('./tools/AudioGenerationTool'))
const VoiceCloningTool = lazy(() => import('./tools/VoiceCloningTool'))
const LipSyncTool = lazy(() => import('./tools/LipSyncTool'))
const ReframeTool = lazy(() => import('./tools/ReframeTool'))
const FaceSwapTool = lazy(() => import('./tools/FaceSwapTool'))
const ComingSoonTool = lazy(() => import('./tools/ComingSoonTool'))
const MyMediaTool = lazy(() => import('./tools/MyMediaTool'))
const Gallery = lazy(() => import('../pages/Gallery'))
const AdminPanelTool = lazy(() => import('./tools/AdminPanelTool'))
const APIKeysTool = lazy(() => import('./tools/APIKeysTool'))
const ProfileTool = lazy(() => import('./tools/ProfileTool'))
const UserProfilePage = lazy(() => import('../pages/UserProfilePage'))
import LogViewer from '../components/LogViewer'
import { sendClientLog } from '../logging'

// Loading fallback component
function ToolLoader() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '3rem',
      color: 'var(--text-muted)',
      gap: '0.75rem',
    }}>
      <Loader2 size={24} className="animate-spin" />
      <span>Loading tool...</span>
    </div>
  )
}

export default function Dashboard() {
  const [activeToolId, setActiveToolId] = useState(TOOL_IDS.IMAGE_TO_VIDEO)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [paramsCollapsed, setParamsCollapsed] = useState(false)

  // Deep-link: ?openItem=<media_id> → switch to gallery
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (params.get('openItem')) {
      setActiveToolId(TOOL_IDS.GALLERY)
    }
  }, [])

  // Mobile parameters panel state
  const [mobileParamsOpen, setMobileParamsOpen] = useState(false)

  // Mobile navigation menu state
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

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

  // Pending import: { item, workflow } - set when user picks "Use in tool" from MyMedia
  const [pendingImport, setPendingImport] = useState(null)

  // User profile view state — userId of profile being viewed
  const [viewingProfile, setViewingProfile] = useState(null)

  // Send media item to a tool for import (component-level so all MyMediaTool instances can use it)
  const handleSendToTool = (toolId, importData) => {
    setPendingImport(importData)
    setActiveToolId(toolId)
  }

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
      case TOOL_IDS.POST_PROCESSING:
        return 'Post-Processing'
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
      case TOOL_IDS.MY_PROFILE:
        return 'My Profile'
      case TOOL_IDS.API_KEYS:
        return 'API Keys'
      case TOOL_IDS.ADMIN_PANEL:
        return 'Admin Panel'
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

    // Wrap tool component in Suspense for lazy loading
    const wrapWithSuspense = (component) => (
      <Suspense fallback={<ToolLoader />}>
        {component}
      </Suspense>
    )

    switch (activeToolId) {
      case TOOL_IDS.TEXT_TO_VIDEO:
        return wrapWithSuspense(<TextToVideoTool onOutput={setOutput} onRefreshHistory={onRefreshHistory} onParamsChange={onParamsChange} onJobSubmitted={onJobSubmitted} pendingImport={pendingImport} onImportConsumed={() => setPendingImport(null)} />)
      case TOOL_IDS.IMAGE_TO_VIDEO:
        return wrapWithSuspense(<ImageToVideoTool onOutput={setOutput} onRefreshHistory={onRefreshHistory} onCreationsModeChange={onCreationsModeChange} onParamsChange={onParamsChange} onJobSubmitted={onJobSubmitted} pendingImport={pendingImport} onImportConsumed={() => setPendingImport(null)} />)
      case TOOL_IDS.TEXT_TO_IMAGE_TO_VIDEO:
        return wrapWithSuspense(<TextToImageToVideoTool onOutput={setOutput} onParamsChange={onParamsChange} onJobSubmitted={onJobSubmitted} />)
      case TOOL_IDS.PIPELINE:
        return wrapWithSuspense(<PipelineTool />)
      case TOOL_IDS.LORA_TRAINING:
        return wrapWithSuspense(<LoRATrainingTool onOutput={setOutput} />)

      case TOOL_IDS.MY_MEDIA_ALL:
        return wrapWithSuspense(<MyMediaTool filter="all" onSendToTool={handleSendToTool} />)
      case TOOL_IDS.MY_MEDIA_VIDEOS:
        return wrapWithSuspense(<MyMediaTool filter="video" onSendToTool={handleSendToTool} />)
      case TOOL_IDS.MY_MEDIA_IMAGES:
        return wrapWithSuspense(<MyMediaTool filter="image" onSendToTool={handleSendToTool} />)
      case TOOL_IDS.MY_MEDIA_AUDIO:
        return wrapWithSuspense(<MyMediaTool filter="audio" onSendToTool={handleSendToTool} />)
      case TOOL_IDS.MY_MEDIA_PROMPTS:
        return wrapWithSuspense(<MyMediaTool filter="prompts" onSendToTool={handleSendToTool} />)
      case TOOL_IDS.GALLERY:
        return wrapWithSuspense(<Gallery onRemix={handleSendToTool} onViewProfile={(userId) => setViewingProfile(userId)} />)

      case TOOL_IDS.TEXT_TO_IMAGE:
        return wrapWithSuspense(<TextToImageTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} pendingImport={pendingImport} onImportConsumed={() => setPendingImport(null)} />)

      case TOOL_IDS.IMAGE_TO_TEXT:
        return wrapWithSuspense(<ImageToTextTool pendingImport={pendingImport} onImportConsumed={() => setPendingImport(null)} />)
      case TOOL_IDS.PROMPT_GENERATOR:
        return wrapWithSuspense(<PromptGeneratorTool />)

      case TOOL_IDS.IMAGE_TO_IMAGE:
        return wrapWithSuspense(<ImageToImageTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} pendingImport={pendingImport} onImportConsumed={() => setPendingImport(null)} />)

      case TOOL_IDS.VIDEO_TO_VIDEO:
        return wrapWithSuspense(<VideoToVideoTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />)
      case TOOL_IDS.VIDEO_TO_TEXT:
        return wrapWithSuspense(<VideoToTextTool />)

      case TOOL_IDS.AUDIO_GENERATION:
        return wrapWithSuspense(<AudioGenerationTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />)

      case TOOL_IDS.VOICE_CLONING:
        return wrapWithSuspense(<VoiceCloningTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />)

      case TOOL_IDS.LIP_SYNC:
        return wrapWithSuspense(<LipSyncTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />)

      case TOOL_IDS.SPEECH_TO_VIDEO:
        return wrapWithSuspense(<SpeechToVideoTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />)

      case TOOL_IDS.REFRAME:
        return wrapWithSuspense(<ReframeTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />)
      case TOOL_IDS.FACE_SWAP:
        return wrapWithSuspense(<FaceSwapTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />)

      case TOOL_IDS.POST_PROCESSING:
        return wrapWithSuspense(<PostProcessingTool onOutput={setOutput} onJobSubmitted={onJobSubmitted} />)

      case TOOL_IDS.API_KEYS:
        return wrapWithSuspense(<APIKeysTool />)

      case TOOL_IDS.MY_PROFILE:
        return wrapWithSuspense(<ProfileTool />)

      case TOOL_IDS.ADMIN_PANEL:
        return wrapWithSuspense(<AdminPanelTool />)

      default:
        return wrapWithSuspense(<ComingSoonTool title={toolTitle} />)
    }
  }

  const { nsfwEnabled, setNsfwEnabled } = useNSFW()
  const { user, isAdult, showLoginModal, loginModalMessage, closeLoginModal } = useAuth()

  return (
    <div className="dashboard-wrapper">
      <div className="dashboard-container">
          {/* Mobile nav overlay */}
          <div
            className={`mobile-nav-overlay ${mobileNavOpen ? 'visible' : ''}`}
            onClick={() => setMobileNavOpen(false)}
          />

          {/* Sidebar - also opens on mobile when nav is open */}
          <div className={`sidebar-wrapper ${mobileNavOpen ? 'mobile-open' : ''}`}>
            <Sidebar
              activeToolId={activeToolId}
              onSelectTool={(id) => {
                setActiveToolId(id)
                setMobileNavOpen(false) // Close nav after selection
              }}
              collapsed={sidebarCollapsed}
              onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
            />
          </div>

      <main className="main-content">
        <div className="top-bar">
          {/* Mobile menu button */}
          <button
            className="mobile-menu-btn"
            onClick={() => setMobileNavOpen(!mobileNavOpen)}
          >
            {mobileNavOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
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
                  const videoUrl = getMediaUrl(job.output_video, job.signed_url)
                  setOutput({
                    kind: 'video',
                    url: videoUrl,
                    backendUrl: videoUrl,
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

        {/* User Profile overlay */}
        {viewingProfile ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {wrapWithSuspense(
              <UserProfilePage
                userId={viewingProfile}
                onBack={() => setViewingProfile(null)}
                onOpenItem={(item) => {
                  // Return to gallery and open the item
                  setViewingProfile(null)
                  setActiveToolId(TOOL_IDS.GALLERY)
                }}
              />
            )}
          </div>
        ) : /* Full-width layout for My Media tools and Gallery */
        (activeToolId === TOOL_IDS.MY_MEDIA_ALL ||
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
            {/* Mobile overlay */}
            <div
              className={`mobile-params-overlay ${mobileParamsOpen ? 'visible' : ''}`}
              onClick={() => setMobileParamsOpen(false)}
            />

            {/* Mobile toggle button */}
            <button
              className={`mobile-params-toggle ${mobileParamsOpen ? 'open' : ''}`}
              onClick={() => setMobileParamsOpen(!mobileParamsOpen)}
            >
              <Settings2 size={18} />
              {mobileParamsOpen ? 'Hide Parameters' : 'Show Parameters'}
              <ChevronUp size={18} />
            </button>

            <section className={`controls-panel ${mobileParamsOpen ? 'mobile-open' : 'mobile-collapsed'} ${paramsCollapsed ? 'collapsed' : ''}`}>
              <div className="panel-header" style={{ marginBottom: paramsCollapsed ? '0' : '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div className="panel-title" style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  {!paramsCollapsed && 'Parameters'}
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  {!paramsCollapsed && (
                    <button
                      className="icon-btn"
                      onClick={handleDownloadParams}
                      title="Download parameters als JSON"
                      style={{ padding: '4px' }}
                    >
                      <Download size={16} />
                    </button>
                  )}
                  <button
                    className="icon-btn panel-collapse-btn"
                    onClick={() => setParamsCollapsed(!paramsCollapsed)}
                    title={paramsCollapsed ? 'Expand parameters' : 'Collapse parameters'}
                    style={{ padding: '4px' }}
                  >
                    {paramsCollapsed ? <PanelRight size={16} /> : <PanelRightClose size={16} />}
                  </button>
                </div>
              </div>
              {!paramsCollapsed && (
                <>
                  <div className="panel-body">{renderControls()}</div>
                  {/* Mobile close button at bottom */}
                  <button
                    className="mobile-close-params"
                    onClick={() => setMobileParamsOpen(false)}
                  >
                    <ChevronUp size={18} />
                    Close Parameters
                  </button>
                </>
              )}
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
                    onSendToTool={handleSendToTool}
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
      {user?.email === 'mark.op.mobiel@gmail.com' && <LogViewer />}
    </div>
  )
}
