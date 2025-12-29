import React, { useState } from 'react'
import { Workflow, ArrowRight, CheckCircle2, Circle, Play } from 'lucide-react'

export default function PipelineTool() {
  const [steps, setSteps] = useState([
    { id: 1, name: 'Text Generation', status: 'completed', description: 'Generate prompt from keywords' },
    { id: 2, name: 'Text to Image', status: 'ready', description: 'Create base image' },
    { id: 3, name: 'Image to Video', status: 'pending', description: 'Animate the image' },
    { id: 4, name: 'Upscale', status: 'pending', description: 'Enhance resolution' },
  ])

  const [activeStep, setActiveStep] = useState(2)

  return (
    <div className="tool-container">
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Production Pipeline</div>
          <Workflow size={16} className="text-muted" />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {steps.map((step, index) => (
            <div 
              key={step.id} 
              className={`pipeline-step ${activeStep === step.id ? 'active' : ''}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '16px',
                padding: '16px',
                backgroundColor: activeStep === step.id ? '#1a1a1a' : 'transparent',
                borderRadius: '8px',
                border: activeStep === step.id ? '1px solid var(--border-color)' : '1px solid transparent',
                opacity: step.status === 'pending' ? 0.5 : 1
              }}
            >
              <div style={{ 
                width: '32px', 
                height: '32px', 
                borderRadius: '50%', 
                backgroundColor: step.status === 'completed' ? '#22c55e' : (activeStep === step.id ? 'var(--text-primary)' : '#333'),
                color: step.status === 'completed' || activeStep === step.id ? 'var(--bg-root)' : 'var(--text-secondary)',
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                fontWeight: 'bold',
                fontSize: '0.9rem'
              }}>
                {step.status === 'completed' ? <CheckCircle2 size={18} /> : step.id}
              </div>
              
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{step.name}</div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{step.description}</div>
              </div>

              {index < steps.length - 1 && (
                <ArrowRight size={16} className="text-muted" style={{ opacity: 0.3 }} />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Step Configuration: {steps.find(s => s.id === activeStep)?.name}</div>
        </div>
        
        <div className="placeholder-state" style={{ padding: '20px 0' }}>
          <div className="text-muted">Configuration options for this step would appear here.</div>
        </div>
      </div>

      <button className="primary-btn" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
        <Play size={18} />
        Run Pipeline
      </button>
    </div>
  )
}
