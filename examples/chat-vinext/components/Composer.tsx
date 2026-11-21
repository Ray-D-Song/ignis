'use client';

import React, { useRef } from 'react';
import { ArrowUp, ImagePlus, Square, Volume2, X } from 'lucide-react';
import { COPY, type UiLang } from '@/lib/i18n';
import type { PendingAudioInput, PendingImageInput } from '@/lib/agent-harness';

interface ComposerProps {
  lang: UiLang;
  mode: 'agent' | 'chat';
  input: string;
  onInputChange: (val: string) => void;
  status: 'idle' | 'loading' | 'ready' | 'generating' | 'error';
  termsBlocked: boolean;
  supportsVision: boolean;
  supportsAudioIn: boolean;
  pendingImage: PendingImageInput | null;
  pendingAudio: PendingAudioInput | null;
  onRemoveImage: () => void;
  onRemoveAudio: () => void;
  onImageFile: (file: File) => void;
  onAudioFile: (file: File) => void;
  onSubmit: () => void;
  onStop: () => void;
}

export function Composer(props: ComposerProps) {
  const {
    lang,
    mode,
    input,
    onInputChange,
    status,
    termsBlocked,
    supportsVision,
    supportsAudioIn,
    pendingImage,
    pendingAudio,
    onRemoveImage,
    onRemoveAudio,
    onImageFile,
    onAudioFile,
    onSubmit,
    onStop,
  } = props;

  const t = COPY[lang];
  const imgRef = useRef<HTMLInputElement | null>(null);
  const audRef = useRef<HTMLInputElement | null>(null);

  const canSend =
    status !== 'loading' &&
    status !== 'generating' &&
    !termsBlocked &&
    Boolean(input.trim() || pendingImage || pendingAudio);

  const placeholder = termsBlocked
    ? t.inputTermsBlocked
    : status === 'ready'
      ? mode === 'agent'
        ? t.inputReadyAgent
        : t.inputReadyChat
      : t.inputWait;

  return (
    <div className="composer-dock-outer">
      <div className="composer-dock">
        {(pendingImage || pendingAudio) && (
          <div className="attach-strip">
            {pendingImage && (
              <span className="attach-pill">
                <img src={pendingImage.previewUrl} alt={pendingImage.name} />
                <span>
                  {pendingImage.name} ({pendingImage.width}×{pendingImage.height})
                </span>
                <button
                  type="button"
                  className="icon-btn"
                  style={{ height: 20, padding: '0 3px' }}
                  onClick={onRemoveImage}
                  aria-label="Remove image"
                >
                  <X size={12} />
                </button>
              </span>
            )}
            {pendingAudio && (
              <span className="attach-pill">
                <Volume2 size={13} />
                <span>
                  {pendingAudio.name} ({pendingAudio.durationSec.toFixed(1)}s)
                </span>
                <button
                  type="button"
                  className="icon-btn"
                  style={{ height: 20, padding: '0 3px' }}
                  onClick={onRemoveAudio}
                  aria-label="Remove audio"
                >
                  <X size={12} />
                </button>
              </span>
            )}
          </div>
        )}

        <textarea
          id="chat-input"
          className="composer-textarea"
          value={input}
          placeholder={placeholder}
          disabled={status === 'loading'}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (canSend) onSubmit();
            }
          }}
        />

        <div className="composer-bar">
          <div className="composer-bar-left">
            {supportsVision && (
              <>
                <input
                  ref={imgRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    e.target.value = '';
                    if (f) onImageFile(f);
                  }}
                />
                <button
                  type="button"
                  className="icon-btn"
                  disabled={status === 'loading' || status === 'generating'}
                  onClick={() => imgRef.current?.click()}
                  title={t.attachImage}
                >
                  <ImagePlus size={14} />
                  <span className="hide-mobile">{t.attachImage}</span>
                </button>
              </>
            )}

            {supportsAudioIn && (
              <>
                <input
                  ref={audRef}
                  type="file"
                  accept="audio/wav,.wav"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    e.target.value = '';
                    if (f) onAudioFile(f);
                  }}
                />
                <button
                  type="button"
                  className="icon-btn"
                  disabled={status === 'loading' || status === 'generating'}
                  onClick={() => audRef.current?.click()}
                  title={t.attachAudio}
                >
                  <Volume2 size={14} />
                  <span className="hide-mobile">{t.attachAudio}</span>
                </button>
              </>
            )}

            <span className="mode-indicator">
              {mode === 'agent' ? t.modeBadgeAgent : t.modeBadgeChat}
            </span>
          </div>

          <div className="composer-bar-right">
            {status === 'generating' ? (
              <button id="btn-stop" type="button" className="stop-cta" onClick={onStop}>
                <Square size={12} fill="currentColor" />
                <span>{t.stop}</span>
              </button>
            ) : (
              <button
                id="btn-send"
                type="button"
                className="send-cta"
                disabled={!canSend}
                onClick={onSubmit}
              >
                <span>{mode === 'agent' ? t.send : t.sendChat}</span>
                <ArrowUp size={13} strokeWidth={2.5} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
