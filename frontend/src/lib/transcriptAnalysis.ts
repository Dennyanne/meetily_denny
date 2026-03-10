import { Transcript, TranscriptSegmentData, TranscriptTopic } from '@/types';

const TOPIC_WINDOW_SECONDS = 180;
const MIN_TOPIC_DURATION_SECONDS = 45;
const MIN_CONTEXT_KEYWORDS = 4;
const MIN_SEGMENT_KEYWORDS = 3;
const TOPIC_SIMILARITY_THRESHOLD = 0.28;
const TOPIC_NOVELTY_THRESHOLD = 0.7;
const CONTEXT_SPLIT_SECONDS = 90;

const STOP_WORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from',
  'get', 'got', 'had', 'has', 'have', 'how', 'i', 'if', 'in', 'into', 'is',
  'it', 'its', 'just', 'kind', 'like', 'me', 'more', 'need', 'of', 'on',
  'or', 'our', 'so', 'some', 'that', 'the', 'their', 'them', 'there', 'they',
  'this', 'to', 'up', 'us', 'was', 'we', 'were', 'what', 'when', 'which',
  'who', 'will', 'with', 'would', 'yeah', 'you', 'your'
]);

const SPEAKER_PREFIX_PATTERNS = [
  /^(speaker\s+\d+)\s*[:\-]\s+/i,
  /^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s*[:\-]\s+/
];

type SegmentInput = Pick<TranscriptSegmentData, 'id' | 'timestamp' | 'endTime' | 'text' | 'confidence' | 'speakerLabel'>;

interface TopicGroup {
  id: string;
  title: string;
  segments: TranscriptSegmentData[];
  startTime: number;
  lastTimestamp: number;
}

function extractKeywords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((token) => token.length >= 3 && !STOP_WORDS.has(token));
}

function buildKeywordFrequency(texts: string[]): Map<string, number> {
  const frequency = new Map<string, number>();

  for (const text of texts) {
    for (const token of extractKeywords(text)) {
      frequency.set(token, (frequency.get(token) ?? 0) + 1);
    }
  }

  return frequency;
}

function getSimilarityScore(left: string[], right: string[]): number {
  if (left.length === 0 || right.length === 0) {
    return 0;
  }

  const leftSet = new Set(left);
  const rightSet = new Set(right);
  let overlap = 0;

  for (const token of leftSet) {
    if (rightSet.has(token)) {
      overlap += 1;
    }
  }

  return overlap / Math.max(leftSet.size, rightSet.size);
}

function getNoveltyScore(segmentKeywords: string[], contextKeywords: string[]): number {
  if (segmentKeywords.length === 0) {
    return 0;
  }

  const contextSet = new Set(contextKeywords);
  const uniqueSegmentKeywords = [...new Set(segmentKeywords)];
  const novelCount = uniqueSegmentKeywords.filter((token) => !contextSet.has(token)).length;

  return novelCount / uniqueSegmentKeywords.length;
}

function getTopicTitle(texts: string[], index: number): string {
  const entries = [...buildKeywordFrequency(texts).entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 3)
    .map(([token]) => token.charAt(0).toUpperCase() + token.slice(1));

  if (entries.length === 0) {
    return `Topic ${index + 1}`;
  }

  return entries.join(' / ');
}

export function extractSpeakerLabel(text: string, explicitLabel?: string): string | undefined {
  if (explicitLabel && explicitLabel.trim()) {
    return explicitLabel.trim();
  }

  const trimmed = text.trim();
  for (const pattern of SPEAKER_PREFIX_PATTERNS) {
    const match = trimmed.match(pattern);
    if (match?.[1]) {
      return match[1].trim();
    }
  }

  return undefined;
}

export function stripSpeakerLabel(text: string, explicitLabel?: string): string {
  const trimmed = text.trim();
  const speakerLabel = extractSpeakerLabel(trimmed, explicitLabel);

  if (!speakerLabel) {
    return trimmed;
  }

  for (const pattern of SPEAKER_PREFIX_PATTERNS) {
    if (pattern.test(trimmed)) {
      return trimmed.replace(pattern, '').trim();
    }
  }

  return trimmed;
}

export function collectMeetingSpeakers(transcripts: Transcript[]): string[] {
  const speakers = new Set<string>();

  for (const transcript of transcripts) {
    const speaker = extractSpeakerLabel(transcript.text, transcript.speaker_label);
    if (speaker) {
      speakers.add(speaker);
    }
  }

  return [...speakers];
}

export function segmentTranscriptSegments(segments: SegmentInput[]): TranscriptSegmentData[] {
  if (segments.length === 0) {
    return [];
  }

  const sortedSegments = [...segments].sort((left, right) => left.timestamp - right.timestamp);
  const topicGroups: TopicGroup[] = [];

  for (const rawSegment of sortedSegments) {
    const speakerLabel = extractSpeakerLabel(rawSegment.text, rawSegment.speakerLabel);
    const cleanedText = stripSpeakerLabel(rawSegment.text, rawSegment.speakerLabel);
    const segment: TranscriptSegmentData = {
      ...rawSegment,
      text: cleanedText,
      speakerLabel,
    };

    const currentGroup = topicGroups[topicGroups.length - 1];
    if (!currentGroup) {
      topicGroups.push({
        id: `topic-1`,
        title: 'Topic 1',
        segments: [segment],
        startTime: segment.timestamp,
        lastTimestamp: segment.timestamp,
      });
      continue;
    }

    const windowSegments = currentGroup.segments.filter(
      (item) => item.timestamp >= segment.timestamp - TOPIC_WINDOW_SECONDS
    );
    const recentSegments = windowSegments.filter(
      (item) => item.timestamp >= segment.timestamp - CONTEXT_SPLIT_SECONDS
    );
    const olderSegments = windowSegments.filter(
      (item) => item.timestamp < segment.timestamp - CONTEXT_SPLIT_SECONDS
    );

    const contextKeywords = extractKeywords(windowSegments.map((item) => item.text).join(' '));
    const recentKeywords = extractKeywords(recentSegments.map((item) => item.text).join(' '));
    const olderKeywords = extractKeywords(olderSegments.map((item) => item.text).join(' '));
    const segmentKeywords = extractKeywords(segment.text);
    const similarity = getSimilarityScore(segmentKeywords, contextKeywords);
    const recentSimilarity = getSimilarityScore(segmentKeywords, recentKeywords);
    const olderSimilarity = getSimilarityScore(segmentKeywords, olderKeywords);
    const novelty = getNoveltyScore(segmentKeywords, contextKeywords);
    const topicAge = segment.timestamp - currentGroup.startTime;

    const shouldStartNewTopic =
      topicAge >= MIN_TOPIC_DURATION_SECONDS &&
      contextKeywords.length >= MIN_CONTEXT_KEYWORDS &&
      segmentKeywords.length >= MIN_SEGMENT_KEYWORDS &&
      (
        similarity < TOPIC_SIMILARITY_THRESHOLD ||
        (recentSimilarity < TOPIC_SIMILARITY_THRESHOLD && novelty >= TOPIC_NOVELTY_THRESHOLD) ||
        (olderKeywords.length >= MIN_CONTEXT_KEYWORDS && recentSimilarity < olderSimilarity * 0.6)
      );

    if (shouldStartNewTopic) {
      topicGroups.push({
        id: `topic-${topicGroups.length + 1}`,
        title: `Topic ${topicGroups.length + 1}`,
        segments: [segment],
        startTime: segment.timestamp,
        lastTimestamp: segment.timestamp,
      });
      continue;
    }

    currentGroup.segments.push(segment);
    currentGroup.lastTimestamp = segment.timestamp;
  }

  topicGroups.forEach((group, index) => {
    group.title = getTopicTitle(group.segments.slice(0, 6).map((segment) => segment.text), index);
  });

  return topicGroups.flatMap((group) =>
    group.segments.map((segment, index) => ({
      ...segment,
      topicId: group.id,
      topicTitle: group.title,
      isTopicStart: index === 0,
    }))
  );
}

export function applyTopicSegmentation(
  segments: SegmentInput[],
  topics?: TranscriptTopic[] | null
): TranscriptSegmentData[] {
  const baseSegments = segmentTranscriptSegments(segments);

  if (!topics || topics.length === 0) {
    return baseSegments;
  }

  const topicBySegmentId = new Map<string, { topicId: string; topicTitle: string }>();

  topics.forEach((topic, index) => {
    const topicId = topic.id || `llm-topic-${index + 1}`;
    topic.segmentIds.forEach((segmentId) => {
      topicBySegmentId.set(segmentId, {
        topicId,
        topicTitle: topic.title,
      });
    });
  });

  let previousTopicId: string | undefined;

  return baseSegments.map((segment) => {
    const llmTopic = topicBySegmentId.get(segment.id);
    const finalSegment = llmTopic
      ? {
          ...segment,
          topicId: llmTopic.topicId,
          topicTitle: llmTopic.topicTitle,
        }
      : segment;

    const isTopicStart = finalSegment.topicId !== previousTopicId;
    previousTopicId = finalSegment.topicId;

    return {
      ...finalSegment,
      isTopicStart,
    };
  });
}

function formatRecordingTime(seconds: number | undefined, fallback: string): string {
  if (seconds === undefined) {
    return fallback;
  }

  const totalSeconds = Math.floor(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const remainder = totalSeconds % 60;
  return `[${minutes.toString().padStart(2, '0')}:${remainder.toString().padStart(2, '0')}]`;
}

export function formatTranscriptForSummary(transcripts: Transcript[]): string {
  const segmented = segmentTranscriptSegments(
    transcripts.map((transcript) => ({
      id: transcript.id,
      timestamp: transcript.audio_start_time ?? 0,
      endTime: transcript.audio_end_time,
      text: transcript.text,
      confidence: transcript.confidence,
      speakerLabel: transcript.speaker_label,
    }))
  );

  let currentTopicId: string | undefined;
  const lines: string[] = [];

  for (const segment of segmented) {
    if (segment.topicId !== currentTopicId) {
      currentTopicId = segment.topicId;
      lines.push(`\n## ${segment.topicTitle}`);
    }

    const prefix = segment.speakerLabel ? `${segment.speakerLabel}: ` : '';
    lines.push(`${formatRecordingTime(segment.timestamp, '[--:--]')} ${prefix}${segment.text}`);
  }

  return lines.join('\n').trim();
}
