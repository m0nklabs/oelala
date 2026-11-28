/** Tests for the shared source-picker card and the embedded picker mode. */
import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Upload, FolderOpen } from 'lucide-react'

vi.mock('../../dashboard/tools/MyMediaTool', () => ({
  default: ({ onSelectItem }) => (
    <div data-testid="media-grid">
      <button onClick={() => onSelectItem({ filename: 'picked.png' })}>pick</button>
    </div>
  ),
}))

import SourcePickerCard from '../SourcePickerCard'
import CreationsPickerModal from '../CreationsPickerModal'

const TABS = [
  { id: 'file', label: 'Upload File', icon: Upload },
  { id: 'creations', label: 'From My Creations', icon: FolderOpen },
]

describe('SourcePickerCard', () => {
  it('renders the title and one tab per entry', () => {
    render(
      <SourcePickerCard title="Source Image" tabs={TABS} active="file" onChange={() => {}}>
        <p>content</p>
      </SourcePickerCard>
    )
    expect(screen.getByText('Source Image')).toBeTruthy()
    expect(screen.getAllByRole('tab')).toHaveLength(2)
    expect(screen.getByText('content')).toBeTruthy()
  })

  it('marks only the active tab selected', () => {
    render(
      <SourcePickerCard tabs={TABS} active="creations" onChange={() => {}}>
        <p />
      </SourcePickerCard>
    )
    const [fileTab, creationsTab] = screen.getAllByRole('tab')
    expect(fileTab.getAttribute('aria-selected')).toBe('false')
    expect(creationsTab.getAttribute('aria-selected')).toBe('true')
    expect(creationsTab.className).toContain('active')
  })

  it('reports the clicked tab id', () => {
    const onChange = vi.fn()
    render(
      <SourcePickerCard tabs={TABS} active="file" onChange={onChange}>
        <p />
      </SourcePickerCard>
    )
    fireEvent.click(screen.getByText('From My Creations'))
    expect(onChange).toHaveBeenCalledWith('creations')
  })

  it('links each tab to the panel it controls', () => {
    render(
      <SourcePickerCard tabs={TABS} active="creations" onChange={() => {}}>
        <p>panel content</p>
      </SourcePickerCard>
    )
    const [, creationsTab] = screen.getAllByRole('tab')
    expect(creationsTab.getAttribute('aria-controls')).toBe('source-tabpanel-creations')
    const panel = screen.getByRole('tabpanel')
    expect(panel.id).toBe('source-tabpanel-creations')
    expect(panel.getAttribute('aria-labelledby')).toBe('source-tab-creations')
    expect(panel.textContent).toContain('panel content')
  })

  it('keeps the inactive tabs out of the tab order', () => {
    render(
      <SourcePickerCard tabs={TABS} active="file" onChange={() => {}}>
        <p />
      </SourcePickerCard>
    )
    const [fileTab, creationsTab] = screen.getAllByRole('tab')
    expect(fileTab.getAttribute('tabindex')).toBe('0')
    expect(creationsTab.getAttribute('tabindex')).toBe('-1')
  })

  it('renders a header action and a badge', () => {
    render(
      <SourcePickerCard
        tabs={[{ id: 'library', label: 'My Uploads', badge: 3 }]}
        active="library"
        onChange={() => {}}
        headerAction={<button>reset</button>}
      >
        <p />
      </SourcePickerCard>
    )
    expect(screen.getByText('reset')).toBeTruthy()
    expect(screen.getByText('(3)')).toBeTruthy()
  })
})

describe('CreationsPickerModal embedded mode', () => {
  it('renders only the grid — no panel header or close button', async () => {
    const { container } = render(
      <CreationsPickerModal show embedded onClose={() => {}} onSelect={() => {}} filter="image" />
    )
    // MyMediaTool is lazy-loaded, so the grid appears after the Suspense resolves.
    expect(await screen.findByTestId('media-grid')).toBeTruthy()
    expect(container.querySelector('.creations-picker-header')).toBeNull()
    expect(container.querySelector('.creations-picker-close')).toBeNull()
  })

  it('keeps the framed mode for callers that still use it', () => {
    const { container } = render(
      <CreationsPickerModal show onClose={() => {}} onSelect={() => {}} title="Pick one" />
    )
    expect(container.querySelector('.creations-picker-header')).toBeTruthy()
    expect(screen.getByText('Pick one')).toBeTruthy()
  })

  it('selects an item and closes', async () => {
    const onSelect = vi.fn()
    const onClose = vi.fn()
    render(<CreationsPickerModal show embedded onClose={onClose} onSelect={onSelect} />)
    fireEvent.click(await screen.findByText('pick'))
    expect(onSelect).toHaveBeenCalledWith({ filename: 'picked.png' })
    expect(onClose).toHaveBeenCalled()
  })

  it('renders nothing when show is false', () => {
    const { container } = render(
      <CreationsPickerModal show={false} embedded onClose={() => {}} onSelect={() => {}} />
    )
    expect(container.firstChild).toBeNull()
  })
})
