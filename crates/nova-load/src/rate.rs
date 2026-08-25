//! Per-process upsert throttle: a token bucket denominated in points.
//!
//! `loader.max_points_per_sec` is an upper bound, not a target — the loader
//! gets as close to it as the store allows and never exceeds it. The bucket
//! holds one second of the rate (or one batch, whichever is larger, so a
//! single batch can always proceed), starts full, and refills continuously.
//! Every upsert *attempt* draws its batch's worth of tokens — retries are
//! load on the store too. Shared by all of a worker's concurrent upsert tasks.
//!
//! Uses `tokio::time::Instant` so tests can run under tokio's paused clock.

use std::time::Duration;

use tokio::time::Instant;

pub struct RateLimiter {
    /// Tokens (points) added per second.
    rate: f64,
    /// Burst ceiling: tokens the bucket can hold.
    capacity: f64,
    state: tokio::sync::Mutex<Bucket>,
}

struct Bucket {
    tokens: f64,
    refilled_at: Instant,
}

impl RateLimiter {
    /// `points_per_sec` is the sustained ceiling (clamped to ≥ 1);
    /// `batch_size` guarantees the bucket can hold at least one batch.
    pub fn new(points_per_sec: u64, batch_size: usize) -> Self {
        let rate = points_per_sec.max(1) as f64;
        let capacity = rate.max(batch_size as f64);
        Self {
            rate,
            capacity,
            state: tokio::sync::Mutex::new(Bucket { tokens: capacity, refilled_at: Instant::now() }),
        }
    }

    /// Wait until `points` tokens are available, then take them. A request
    /// larger than the bucket is clamped to its capacity rather than blocking
    /// forever — it just drains the bucket.
    pub async fn acquire(&self, points: usize) {
        let need = (points as f64).min(self.capacity);
        loop {
            let wait = {
                let mut b = self.state.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(b.refilled_at).as_secs_f64();
                b.tokens = (b.tokens + elapsed * self.rate).min(self.capacity);
                b.refilled_at = now;
                if b.tokens >= need {
                    b.tokens -= need;
                    return;
                }
                Duration::from_secs_f64((need - b.tokens) / self.rate)
            };
            tokio::time::sleep(wait).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Under tokio's paused clock, sleeps advance virtual time instantly, so
    /// elapsed time measures exactly how long the limiter *would* have held
    /// the caller.
    #[tokio::test(start_paused = true)]
    async fn sustained_rate_is_capped() {
        let limiter = RateLimiter::new(100, 10);
        let start = Instant::now();

        // The bucket starts full: one second's worth is free.
        limiter.acquire(100).await;
        assert_eq!(start.elapsed(), Duration::ZERO);

        // The next 100 need a full second of refill.
        limiter.acquire(100).await;
        assert!(start.elapsed() >= Duration::from_millis(999), "{:?}", start.elapsed());

        // 500 more points ≈ 5 more seconds, regardless of batch shape.
        for _ in 0..50 {
            limiter.acquire(10).await;
        }
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(5_990), "{elapsed:?}");
        assert!(elapsed < Duration::from_millis(6_100), "{elapsed:?}");
    }

    /// A batch bigger than the bucket drains it and proceeds — it must never
    /// deadlock waiting for tokens that can't exist.
    #[tokio::test(start_paused = true)]
    async fn oversized_batch_never_deadlocks() {
        let limiter = RateLimiter::new(10, 5); // capacity 10
        let start = Instant::now();
        limiter.acquire(1_000).await; // clamps to 10: instant
        assert_eq!(start.elapsed(), Duration::ZERO);
        limiter.acquire(1_000).await; // clamps to 10: one second of refill
        assert!(start.elapsed() >= Duration::from_millis(999), "{:?}", start.elapsed());
    }

    /// The capacity is at least one batch even when the rate is tiny.
    #[tokio::test(start_paused = true)]
    async fn capacity_covers_one_batch() {
        let limiter = RateLimiter::new(1, 256);
        let start = Instant::now();
        limiter.acquire(256).await; // full batch from the initial fill: instant
        assert_eq!(start.elapsed(), Duration::ZERO);
    }
}
