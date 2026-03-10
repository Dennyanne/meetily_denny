'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useConfig } from '@/contexts/ConfigContext';
import { applyTopicSegmentation } from '@/lib/transcriptAnalysis';
import { TopicSegmentationResponse, Transcript, TranscriptSegmentData } from '@/types';

const LIVE_ANALYSIS_WINDOW_MS = 12 * 60 * 1000;
const ANALYSIS_DEBOUNCE_MS = 8000;
const MIN_SEGMENTS_FOR_LLM = 6;
const MAX_STATIC_ANALYSIS_SEGMENTS = 90;
const MIN_NEW_SEGMENTS_BEFORE_REFRESH = 3;

interface TopicSegmentationSegmentInput {
  id: string;
  startMs: number;
  endMs?: number;
  text: string;
  speakerLabel?: string;
}

function toSegmentInputs(transcripts: Transcript[]): TopicSegmentationSegmentInput[] {
  return transcripts
    .filter((transcript) => !transcript.is_partial)
    .map((transcript) => ({
      id: transcript.id,
      startMs: Math.max(0, Math.round((transcript.audio_start_time ?? 0) * 1000)),
      endMs: transcript.audio_end_time !== undefined
        ? Math.max(0, Math.round(transcript.audio_end_time * 1000))
        : undefined,
      text: transcript.text,
      speakerLabel: transcript.speaker_label,
    }));
}

function selectWindow(transcripts: Transcript[], isRecording: boolean): Transcript[] {
  const finalized = transcripts.filter((transcript) => !transcript.is_partial);
  if (!isRecording || finalized.length === 0) {
    return finalized.slice(-MAX_STATIC_ANALYSIS_SEGMENTS);
  }

  const latestStart = finalized[finalized.length - 1].audio_start_time ?? 0;
  const cutoffSeconds = Math.max(0, latestStart - LIVE_ANALYSIS_WINDOW_MS / 1000);
  return finalized.filter((transcript) => (transcript.audio_start_time ?? 0) >= cutoffSeconds);
}

export function useTopicSegmentation(
  transcripts: Transcript[],
  isRecording: boolean,
  meetingId?: string | null
): { segments: TranscriptSegmentData[]; isUsingLlmTopics: boolean } {
  const { modelConfig } = useConfig();
  const [topicResponse, setTopicResponse] = useState<TopicSegmentationResponse | null>(null);
  const [hasLoadedPersistedTopics, setHasLoadedPersistedTopics] = useState(false);
  const lastSignatureRef = useRef<string>('');
  const lastErrorAtRef = useRef<number>(0);
  const lastSavedSignatureRef = useRef<string>('');

  useEffect(() => {
    setTopicResponse(null);
    setHasLoadedPersistedTopics(false);
    lastSignatureRef.current = '';
    lastSavedSignatureRef.current = '';
  }, [meetingId]);

  useEffect(() => {
    if (!meetingId || hasLoadedPersistedTopics) {
      return;
    }

    let isCancelled = false;

    const loadPersistedTopics = async () => {
      try {
        const response = await invoke<{ data: TopicSegmentationResponse | null }>('api_get_topic_segmentation', {
          meetingId,
        });
        if (!isCancelled && response.data) {
          setTopicResponse(response.data);
        }
      } catch (error) {
        console.warn('[TopicSegmentation] Failed to load saved topic segmentation:', error);
      } finally {
        if (!isCancelled) {
          setHasLoadedPersistedTopics(true);
        }
      }
    };

    loadPersistedTopics();

    return () => {
      isCancelled = true;
    };
  }, [hasLoadedPersistedTopics, meetingId]);

  const analyzedWindow = useMemo(
    () => selectWindow(transcripts, isRecording),
    [transcripts, isRecording]
  );

  useEffect(() => {
    const segmentInputs = toSegmentInputs(analyzedWindow);
    if (segmentInputs.length < MIN_SEGMENTS_FOR_LLM) {
      return;
    }

    if (topicResponse && segmentInputs.length - topicResponse.analyzedSegmentCount < MIN_NEW_SEGMENTS_BEFORE_REFRESH) {
      return;
    }

    const signature = [
      modelConfig.provider,
      modelConfig.model,
      segmentInputs.length,
      segmentInputs[0]?.id,
      segmentInputs[segmentInputs.length - 1]?.id,
    ].join(':');

    if (signature === lastSignatureRef.current) {
      return;
    }

    const timeout = window.setTimeout(async () => {
      try {
        const response = await invoke<TopicSegmentationResponse>('api_segment_topics', {
          segments: segmentInputs,
          model: modelConfig.provider,
          modelName: modelConfig.model,
        });
        setTopicResponse(response);
        lastSignatureRef.current = signature;

        if (meetingId) {
          await invoke('api_save_topic_segmentation', {
            meetingId,
            segmentation: response,
          });
          lastSavedSignatureRef.current = signature;
        }
      } catch (error) {
        const now = Date.now();
        if (now - lastErrorAtRef.current > 30000) {
          console.warn('[TopicSegmentation] LLM topic analysis failed, using heuristic fallback:', error);
          lastErrorAtRef.current = now;
        }
      }
    }, ANALYSIS_DEBOUNCE_MS);

    return () => window.clearTimeout(timeout);
  }, [analyzedWindow, meetingId, modelConfig.model, modelConfig.provider, topicResponse]);

  const segments = useMemo(() => {
    const segmentInputs = transcripts.map((transcript) => ({
      id: transcript.id,
      timestamp: transcript.audio_start_time ?? 0,
      endTime: transcript.audio_end_time,
      text: transcript.text,
      confidence: transcript.confidence,
      speakerLabel: transcript.speaker_label,
    }));

    return applyTopicSegmentation(segmentInputs, topicResponse?.topics ?? null);
  }, [topicResponse?.topics, transcripts]);

  return {
    segments,
    isUsingLlmTopics: Boolean(topicResponse?.topics?.length),
  };
}
