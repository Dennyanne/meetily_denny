use realfft::RealFftPlanner;

const MIN_SAMPLES: usize = 8_000;
const FFT_SIZE: usize = 1024;
const HOP_SIZE: usize = 512;
const BAND_COUNT: usize = 12;
const MATCH_THRESHOLD: f32 = 0.72;
const NEW_SPEAKER_THRESHOLD: f32 = 0.68;
const PITCH_MATCH_THRESHOLD: f32 = 0.08;

#[derive(Debug, Clone)]
struct SpeakerProfile {
    label: String,
    centroid: Vec<f32>,
    pitch: f32,
    observations: usize,
}

#[derive(Debug, Default)]
pub struct SpeakerDiarizer {
    speakers: Vec<SpeakerProfile>,
}

impl SpeakerDiarizer {
    pub fn new() -> Self {
        Self {
            speakers: Vec::new(),
        }
    }

    pub fn identify_speaker(&mut self, samples: &[f32], sample_rate: u32) -> Option<String> {
        let pitch = estimate_pitch(samples, sample_rate);
        let embedding = compute_embedding(samples, sample_rate, pitch)?;

        if self.speakers.is_empty() {
            let label = String::from("Speaker 1");
            self.speakers.push(SpeakerProfile {
                label: label.clone(),
                centroid: embedding,
                pitch,
                observations: 1,
            });
            return Some(label);
        }

        let mut best_index = 0usize;
        let mut best_score = f32::MIN;
        let mut best_pitch_distance = f32::MAX;

        for (index, speaker) in self.speakers.iter().enumerate() {
            let pitch_distance = (pitch - speaker.pitch).abs();
            let score = cosine_similarity(&embedding, &speaker.centroid) - pitch_distance * 2.5;
            if score > best_score {
                best_index = index;
                best_score = score;
                best_pitch_distance = pitch_distance;
            }
        }

        if (best_score >= MATCH_THRESHOLD && best_pitch_distance <= PITCH_MATCH_THRESHOLD)
            || best_score >= NEW_SPEAKER_THRESHOLD && self.speakers.len() >= 4
        {
            let speaker = &mut self.speakers[best_index];
            merge_centroid(&mut speaker.centroid, &embedding, speaker.observations);
            speaker.pitch = (speaker.pitch * speaker.observations as f32 + pitch)
                / (speaker.observations as f32 + 1.0);
            speaker.observations += 1;
            return Some(speaker.label.clone());
        }

        let label = format!("Speaker {}", self.speakers.len() + 1);
        self.speakers.push(SpeakerProfile {
            label: label.clone(),
            centroid: embedding,
            pitch,
            observations: 1,
        });
        Some(label)
    }
}

fn compute_embedding(samples: &[f32], sample_rate: u32, pitch: f32) -> Option<Vec<f32>> {
    if samples.len() < MIN_SAMPLES || sample_rate == 0 {
        return None;
    }

    let rms =
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < 0.01 {
        return None;
    }

    let zero_crossings = samples
        .windows(2)
        .filter(|window| {
            (window[0] >= 0.0 && window[1] < 0.0) || (window[0] < 0.0 && window[1] >= 0.0)
        })
        .count() as f32
        / samples.len() as f32;

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(FFT_SIZE);
    let mut input = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();
    let window = hann_window(FFT_SIZE);
    let mut band_accumulator = vec![0.0f32; BAND_COUNT];
    let mut centroid_accumulator = 0.0f32;
    let mut rolloff_accumulator = 0.0f32;
    let mut peak_accumulator = 0.0f32;
    let mut frame_count = 0usize;

    for frame in samples.windows(FFT_SIZE).step_by(HOP_SIZE).take(48) {
        for (index, value) in frame.iter().enumerate() {
            input[index] = *value * window[index];
        }

        if r2c.process(&mut input, &mut spectrum).is_err() {
            return None;
        }

        let magnitudes: Vec<f32> = spectrum.iter().map(|bin| bin.norm_sqr().sqrt()).collect();
        let total_energy: f32 = magnitudes.iter().sum();
        if total_energy <= f32::EPSILON {
            continue;
        }

        let band_size = (magnitudes.len() / BAND_COUNT).max(1);
        for (index, magnitude) in magnitudes.iter().enumerate() {
            let band_index = (index / band_size).min(BAND_COUNT - 1);
            band_accumulator[band_index] += *magnitude;
            centroid_accumulator += *magnitude * index as f32;
        }

        if let Some((peak_index, _)) = magnitudes.iter().enumerate().max_by(|left, right| {
            left.1
                .partial_cmp(right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            peak_accumulator += peak_index as f32 / magnitudes.len() as f32;
        }

        let mut cumulative = 0.0f32;
        for (index, magnitude) in magnitudes.iter().enumerate() {
            cumulative += *magnitude;
            if cumulative / total_energy >= 0.85 {
                rolloff_accumulator += index as f32 / magnitudes.len() as f32;
                break;
            }
        }

        frame_count += 1;
    }

    if frame_count == 0 {
        return None;
    }

    let mut embedding = Vec::with_capacity(BAND_COUNT + 4);
    for band in band_accumulator {
        embedding.push(band / frame_count as f32);
    }
    embedding.push(centroid_accumulator / frame_count as f32 / FFT_SIZE as f32);
    embedding.push(rolloff_accumulator / frame_count as f32);
    embedding.push(peak_accumulator / frame_count as f32);
    embedding.push(zero_crossings);
    embedding.push(rms);
    embedding.push(pitch);
    normalize(&mut embedding);
    Some(embedding)
}

fn estimate_pitch(samples: &[f32], sample_rate: u32) -> f32 {
    let min_lag = (sample_rate / 350).max(1) as usize;
    let max_lag = (sample_rate / 80).max((min_lag + 1) as u32) as usize;
    let analysis_len = samples.len().min(4096);

    let mut best_lag = min_lag;
    let mut best_score = f32::MIN;

    for lag in min_lag..max_lag.min(analysis_len.saturating_sub(1)) {
        let mut score = 0.0f32;
        for index in 0..(analysis_len - lag) {
            score += samples[index] * samples[index + lag];
        }
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    let hz = sample_rate as f32 / best_lag as f32;
    (hz / 400.0).min(1.0)
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| {
            let ratio = index as f32 / size as f32;
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * ratio).cos()
        })
        .collect()
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn merge_centroid(centroid: &mut [f32], sample: &[f32], observations: usize) {
    let weight = observations as f32;
    for (value, new_value) in centroid.iter_mut().zip(sample.iter()) {
        *value = (*value * weight + *new_value) / (weight + 1.0);
    }
    normalize(centroid);
}

#[cfg(test)]
mod tests {
    use super::SpeakerDiarizer;

    fn make_voice_like_samples(freq_a: f32, freq_b: f32) -> Vec<f32> {
        let sample_rate = 16_000.0;
        let duration_seconds = 1.6;
        let total_samples = (sample_rate * duration_seconds) as usize;

        (0..total_samples)
            .map(|index| {
                let t = index as f32 / sample_rate;
                let envelope = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 2.0 * t).sin().abs();
                envelope
                    * ((2.0 * std::f32::consts::PI * freq_a * t).sin() * 0.7
                        + (2.0 * std::f32::consts::PI * freq_b * t).sin() * 0.3)
            })
            .collect()
    }

    fn make_distinct_voice_like_samples() -> Vec<f32> {
        let sample_rate = 16_000.0;
        let duration_seconds = 1.6;
        let total_samples = (sample_rate * duration_seconds) as usize;

        (0..total_samples)
            .map(|index| {
                let t = index as f32 / sample_rate;
                let base = (2.0 * std::f32::consts::PI * 260.0 * t).sin();
                let harmonic = (2.0 * std::f32::consts::PI * 780.0 * t).sin() * 0.45;
                let pulse = if (2.0 * std::f32::consts::PI * 130.0 * t).sin() >= 0.0 {
                    0.25
                } else {
                    -0.25
                };
                let envelope = 0.55 + 0.45 * (2.0 * std::f32::consts::PI * 3.0 * t).sin().abs();
                envelope * (base * 0.55 + harmonic + pulse)
            })
            .collect()
    }

    #[test]
    fn assigns_stable_labels_for_similar_voice_chunks() {
        let mut diarizer = SpeakerDiarizer::new();
        let speaker_a1 = make_voice_like_samples(180.0, 360.0);
        let speaker_a2 = make_voice_like_samples(185.0, 355.0);

        let first = diarizer.identify_speaker(&speaker_a1, 16_000);
        let second = diarizer.identify_speaker(&speaker_a2, 16_000);

        assert_eq!(first, Some(String::from("Speaker 1")));
        assert_eq!(second, Some(String::from("Speaker 1")));
    }

    #[test]
    fn assigns_distinct_labels_for_different_voice_chunks() {
        let mut diarizer = SpeakerDiarizer::new();
        let speaker_a = make_voice_like_samples(170.0, 320.0);
        let speaker_b = make_distinct_voice_like_samples();

        let first = diarizer.identify_speaker(&speaker_a, 16_000);
        let second = diarizer.identify_speaker(&speaker_b, 16_000);

        assert_eq!(first, Some(String::from("Speaker 1")));
        assert_eq!(second, Some(String::from("Speaker 2")));
    }
}
