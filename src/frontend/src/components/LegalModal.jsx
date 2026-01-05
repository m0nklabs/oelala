/**
 * Legal Modal
 * Displays Privacy Policy, Terms of Service, or DMCA Policy
 */

import React, { useState, useEffect } from 'react'
import { X, Shield, FileText, Scale, ExternalLink } from 'lucide-react'

// Legal content embedded as constants
// In production, you might fetch these from markdown files or an API

const PRIVACY_POLICY = `
# Privacy Policy

**Last updated: January 5, 2026**

oelala.xyz ("we", "our", or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, and share information about you when you use our AI image and video generation service.

## 1. Information We Collect

### Account Information
- Email address (for authentication)
- Display name (optional)

### Generated Content
- Images and videos you generate using our service
- Prompts and settings used for generation
- Generation history and favorites

### Usage Data
- Features used and generation counts
- Credit usage and transaction history
- Device type and browser information

### Payment Information
When you purchase credits, payment processing is handled by Stripe. We do not store your credit card details.

## 2. How We Use Your Information

We use your information to:
- Provide and improve our AI generation service
- Process credit purchases and manage your account
- Send service-related notifications
- Prevent abuse and enforce our Terms of Service
- Comply with legal obligations

## 3. Data Storage

- **Authentication**: Supabase (EU region)
- **Generated Media**: Our own servers (Netherlands)
- **Payments**: Stripe

### Data Retention
| Data Type | Retention Period |
|-----------|------------------|
| Account data | Until account deletion |
| Generated media (Free tier) | 30 days |
| Generated media (Paid tier) | Per subscription terms |

## 4. Your Rights (GDPR)

As an EU-based service, you have the right to:
- **Access**: Request a copy of your data
- **Rectification**: Correct inaccurate data
- **Erasure**: Delete your account and data
- **Portability**: Export your data
- **Object**: Opt out of certain processing

To exercise these rights, email: **privacy@oelala.xyz**

## 5. Cookies

We use essential cookies only for authentication and preferences. We do not use tracking or advertising cookies.

## 6. Contact Us

For privacy-related questions: **privacy@oelala.xyz**
`

const TERMS_OF_SERVICE = `
# Terms of Service

**Last updated: January 5, 2026**

By using oelala.xyz ("Service"), you agree to these Terms. Please read them carefully.

## 1. Eligibility

You must be at least 18 years old to use this Service.

## 2. Account Security

You are responsible for maintaining the security of your account and for all activities under your account.

## 3. Credits and Payments

- Generations require credits
- Credits are purchased in packages
- Credits are non-refundable except as required by law
- Unused credits may expire per your subscription tier

## 4. Acceptable Use

### You May NOT Generate:
- Child sexual abuse material (CSAM) – **zero tolerance**
- Non-consensual intimate imagery of real people
- Content that infringes intellectual property
- Content promoting violence, terrorism, or hate
- Fraudulent deepfakes

### We Reserve the Right To:
- Remove violating content
- Suspend or terminate accounts
- Report illegal content to authorities

## 5. Intellectual Property

You retain ownership of content you generate, subject to AI model licenses.

## 6. Disclaimer

THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES. We do not guarantee uninterrupted service or accuracy of AI-generated content.

## 7. Limitation of Liability

Our total liability is limited to the amount you paid us in the past 12 months.

## 8. Governing Law

These Terms are governed by the laws of the Netherlands.

## 9. Contact

For questions: **legal@oelala.xyz**

*By using oelala.xyz, you agree to be bound by these Terms.*
`

const DMCA_POLICY = `
# DMCA Policy

**Last updated: January 5, 2026**

oelala.xyz respects intellectual property rights. This policy outlines our procedures for handling copyright infringement claims.

## Reporting Copyright Infringement

Send a written notice to **dmca@oelala.xyz** containing:

1. Your contact information
2. Identification of the copyrighted work
3. URL of the infringing content
4. Statement of good faith belief
5. Statement of accuracy (under penalty of perjury)
6. Your signature

## Our Response

Upon receiving a valid DMCA notice, we will:
1. Remove or disable access to the content
2. Notify the user who posted the content
3. Provide opportunity for counter-notification

## Counter-Notification

If you believe your content was removed in error, submit a counter-notification to **dmca@oelala.xyz** with:
1. Your contact information
2. Identification of the removed content
3. Statement of good faith belief
4. Consent to jurisdiction
5. Your signature

## Repeat Infringer Policy

- First offense: Warning and content removal
- Second offense: Account suspension
- Third offense: Permanent termination

## AI-Generated Content Note

AI-generated content may incorporate learned patterns from training data. If you believe content infringes your specific copyrighted work, include detailed information about your original work.
`

const CONTENT = {
  privacy: {
    title: 'Privacy Policy',
    icon: Shield,
    content: PRIVACY_POLICY,
  },
  terms: {
    title: 'Terms of Service',
    icon: FileText,
    content: TERMS_OF_SERVICE,
  },
  dmca: {
    title: 'DMCA Policy',
    icon: Scale,
    content: DMCA_POLICY,
  },
}

// Simple markdown-like renderer
function renderContent(text) {
  const lines = text.trim().split('\n')
  const elements = []
  let inTable = false
  let tableRows = []
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    // Table handling
    if (line.startsWith('|')) {
      if (!inTable) {
        inTable = true
        tableRows = []
      }
      if (!line.includes('---')) {
        tableRows.push(line.split('|').filter(c => c.trim()))
      }
      continue
    } else if (inTable) {
      // End of table
      elements.push(
        <table key={`table-${i}`} style={styles.table}>
          <thead>
            <tr>
              {tableRows[0]?.map((cell, j) => (
                <th key={j} style={styles.th}>{cell.trim()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableRows.slice(1).map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => (
                  <td key={ci} style={styles.td}>{cell.trim()}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )
      inTable = false
      tableRows = []
    }
    
    // Headers
    if (line.startsWith('# ')) {
      elements.push(<h1 key={i} style={styles.h1}>{line.slice(2)}</h1>)
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={i} style={styles.h2}>{line.slice(3)}</h2>)
    } else if (line.startsWith('### ')) {
      elements.push(<h3 key={i} style={styles.h3}>{line.slice(4)}</h3>)
    }
    // List items
    else if (line.startsWith('- ')) {
      elements.push(<li key={i} style={styles.li}>{renderInline(line.slice(2))}</li>)
    }
    // Bold paragraph (like **Last updated...**)
    else if (line.startsWith('**') && line.endsWith('**')) {
      elements.push(<p key={i} style={styles.bold}>{line.slice(2, -2)}</p>)
    }
    // Regular paragraph
    else if (line.trim()) {
      elements.push(<p key={i} style={styles.p}>{renderInline(line)}</p>)
    }
    // Empty line
    else {
      elements.push(<div key={i} style={{ height: '8px' }} />)
    }
  }
  
  return elements
}

function renderInline(text) {
  // Handle bold text
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}

const styles = {
  h1: { fontSize: '24px', fontWeight: 'bold', marginBottom: '16px', color: '#fff' },
  h2: { fontSize: '18px', fontWeight: '600', marginTop: '24px', marginBottom: '12px', color: '#e5e7eb' },
  h3: { fontSize: '15px', fontWeight: '600', marginTop: '16px', marginBottom: '8px', color: '#d1d5db' },
  p: { fontSize: '14px', lineHeight: '1.6', marginBottom: '8px', color: '#9ca3af' },
  bold: { fontSize: '14px', fontWeight: '600', marginBottom: '16px', color: '#9ca3af' },
  li: { fontSize: '14px', lineHeight: '1.6', marginLeft: '16px', marginBottom: '4px', color: '#9ca3af' },
  table: { width: '100%', borderCollapse: 'collapse', marginBottom: '16px' },
  th: { padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #374151', color: '#e5e7eb', fontSize: '13px' },
  td: { padding: '8px 12px', borderBottom: '1px solid #1f2937', color: '#9ca3af', fontSize: '13px' },
}

export default function LegalModal({ type = 'privacy', onClose }) {
  const content = CONTENT[type] || CONTENT.privacy
  const Icon = content.icon
  
  // Handle escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [onClose])
  
  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          zIndex: 9998,
        }}
      />
      
      {/* Modal */}
      <div
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '90%',
          maxWidth: '700px',
          maxHeight: '85vh',
          background: '#111827',
          borderRadius: '12px',
          border: '1px solid #374151',
          zIndex: 9999,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '16px 20px',
            borderBottom: '1px solid #374151',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Icon size={20} style={{ color: '#8b5cf6' }} />
            <span style={{ color: '#fff', fontWeight: '600', fontSize: '16px' }}>
              {content.title}
            </span>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              padding: '4px',
              display: 'flex',
            }}
          >
            <X size={20} style={{ color: '#6b7280' }} />
          </button>
        </div>
        
        {/* Content */}
        <div
          style={{
            flex: 1,
            overflow: 'auto',
            padding: '20px',
          }}
        >
          {renderContent(content.content)}
        </div>
        
        {/* Footer */}
        <div
          style={{
            padding: '12px 20px',
            borderTop: '1px solid #374151',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ display: 'flex', gap: '16px' }}>
            {type !== 'privacy' && (
              <button
                onClick={() => window.dispatchEvent(new CustomEvent('showLegal', { detail: 'privacy' }))}
                style={linkStyle}
              >
                Privacy
              </button>
            )}
            {type !== 'terms' && (
              <button
                onClick={() => window.dispatchEvent(new CustomEvent('showLegal', { detail: 'terms' }))}
                style={linkStyle}
              >
                Terms
              </button>
            )}
            {type !== 'dmca' && (
              <button
                onClick={() => window.dispatchEvent(new CustomEvent('showLegal', { detail: 'dmca' }))}
                style={linkStyle}
              >
                DMCA
              </button>
            )}
          </div>
          <button
            onClick={onClose}
            style={{
              background: '#374151',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              padding: '8px 16px',
              fontSize: '14px',
              cursor: 'pointer',
            }}
          >
            Close
          </button>
        </div>
      </div>
    </>
  )
}

const linkStyle = {
  background: 'transparent',
  border: 'none',
  color: '#8b5cf6',
  fontSize: '13px',
  cursor: 'pointer',
  textDecoration: 'underline',
}

// Export a hook to easily show legal modals
export function useLegalModal() {
  const [modalType, setModalType] = useState(null)
  
  useEffect(() => {
    const handler = (e) => setModalType(e.detail)
    window.addEventListener('showLegal', handler)
    return () => window.removeEventListener('showLegal', handler)
  }, [])
  
  return {
    modalType,
    showPrivacy: () => setModalType('privacy'),
    showTerms: () => setModalType('terms'),
    showDMCA: () => setModalType('dmca'),
    close: () => setModalType(null),
  }
}
