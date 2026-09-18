// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Micro-benchmark for the modular-mode weighted (self-correcting) predictor.
//!
//! It drives `WeightedPredictorState::predict_and_property` + `update_errors` over a synthetic
//! image the same way the decoder does (one predict + one error update per sample, left to right,
//! top to bottom), so it directly measures the hot path touched when hardening that code.
//!
//! Requires `--features bench_internals` to reach the otherwise-private predictor module.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use jxl::frame::modular::predict::{PredictionData, WeightedPredictorState};
use jxl::headers::modular::WeightedHeader;

/// The all-default weighted-predictor header (values match the bitstream defaults).
fn default_wp_header() -> WeightedHeader {
    WeightedHeader {
        all_default: true,
        p1c: 16,
        p2c: 10,
        p3ca: 7,
        p3cb: 7,
        p3cc: 7,
        p3cd: 0,
        p3ce: 0,
        w0: 0xd,
        w1: 0xc,
        w2: 0xc,
        w3: 0xc,
    }
}

#[inline(always)]
fn next(state: &mut u64) -> u64 {
    // xorshift64: cheap, deterministic.
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Precomputes one row of neighbour data and correction values so the timed loop spends its time
/// in the predictor rather than in input generation.
fn make_inputs(xsize: usize) -> (Vec<PredictionData>, Vec<i32>) {
    let mut rng = 0x1234_5678_9abc_def1u64;
    let sample = |rng: &mut u64| (next(rng) % 512) as i32 - 256;
    let data = (0..xsize)
        .map(|_| PredictionData {
            top: sample(&mut rng),
            left: sample(&mut rng),
            topright: sample(&mut rng),
            topleft: sample(&mut rng),
            toptop: sample(&mut rng),
            leftleft: sample(&mut rng),
            toprightright: sample(&mut rng),
        })
        .collect();
    let vals = (0..xsize).map(|_| sample(&mut rng)).collect();
    (data, vals)
}

/// Simulates decoding an `xsize` x `ysize` weighted-predicted channel.
fn run(xsize: usize, ysize: usize, data_row: &[PredictionData], vals: &[i32]) -> (i64, i32) {
    let header = default_wp_header();
    let mut state = WeightedPredictorState::new(&header, xsize);
    let mut acc = (0i64, 0i32);
    for y in 0..ysize {
        for x in 0..xsize {
            let (pred, prop) = state.predict_and_property((x, y), &data_row[x]);
            state.update_errors(vals[x], (x, y));
            acc.0 ^= pred;
            acc.1 ^= prop;
        }
    }
    acc
}

fn predict_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_predictor");
    for (xsize, ysize) in [(256usize, 256usize), (512, 512), (1024, 256)] {
        let (data, vals) = make_inputs(xsize);
        group.throughput(Throughput::Elements((xsize * ysize) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{xsize}x{ysize}")),
            &(xsize, ysize),
            |b, &(xs, ys)| {
                b.iter(|| black_box(run(xs, ys, black_box(&data), black_box(&vals))));
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(30);
    targets = predict_benches
);
criterion_main!(benches);
