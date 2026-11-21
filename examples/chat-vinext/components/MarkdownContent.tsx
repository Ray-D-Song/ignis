'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Check, Copy, Play } from 'lucide-react';

interface MarkdownContentProps {
  content: string;
  onPreviewHtml?: (title: string, html: string) => void;
}

export function MarkdownContent({ content, onPreviewHtml }: MarkdownContentProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  return (
    <div className="answer-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre({ children }) {
            const childArr = React.Children.toArray(children);
            const first = childArr[0];
            let lang = 'text';
            let codeText = '';

            if (React.isValidElement<{ className?: string; children?: React.ReactNode }>(first)) {
              const cls = first.props.className || '';
              const match = /language-([^\s]+)/.exec(cls);
              if (match) lang = match[1];
              codeText = String(first.props.children ?? '').replace(/\n$/, '');
            } else {
              codeText = String(children ?? '').replace(/\n$/, '');
            }

            const isPreviewable = lang === 'html' || lang === 'svg';
            const copyKey = `${lang}:${codeText.slice(0, 32)}`;

            return (
              <div className="md-code-wrap">
                <div className="md-code-bar">
                  <span>{lang}</span>
                  <div className="md-code-actions">
                    {isPreviewable && onPreviewHtml && (
                      <button
                        type="button"
                        className="icon-btn"
                        style={{ height: 22, padding: '0 7px', fontSize: 11 }}
                        onClick={() => onPreviewHtml(`${lang.toUpperCase()} Snippet`, codeText)}
                      >
                        <Play size={11} />
                        <span>Preview</span>
                      </button>
                    )}
                    <button
                      type="button"
                      className="icon-btn"
                      style={{ height: 22, padding: '0 7px', fontSize: 11 }}
                      onClick={() => {
                        navigator.clipboard?.writeText(codeText).catch(() => {});
                        setCopiedId(copyKey);
                        setTimeout(() => setCopiedId(null), 1400);
                      }}
                    >
                      {copiedId === copyKey ? <Check size={11} /> : <Copy size={11} />}
                      <span>{copiedId === copyKey ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>
                </div>
                <pre>
                  <code>{codeText}</code>
                </pre>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
