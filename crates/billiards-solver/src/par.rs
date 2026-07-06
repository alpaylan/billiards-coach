//! Parallel-iterator facade: rayon on native, a sequential shim on wasm.
//!
//! The solver's hot paths are written against rayon's `into_par_iter` /
//! `par_iter` / `flat_map_iter`. On `wasm32` (no threads without COOP/COEP +
//! wasm-bindgen-rayon) the same call-sites compile against these drop-in
//! sequential equivalents, so the algorithm code stays identical. Determinism
//! is unaffected — the parallel version already preserves order.

#[cfg(not(target_arch = "wasm32"))]
pub use rayon::prelude::*;

#[cfg(target_arch = "wasm32")]
mod seq {
    /// `into_par_iter()` -> the plain owned iterator.
    pub trait IntoParIter: IntoIterator + Sized {
        fn into_par_iter(self) -> Self::IntoIter {
            self.into_iter()
        }
    }
    impl<T: IntoIterator> IntoParIter for T {}

    /// `par_iter()` on slices/Vecs -> the plain shared iterator.
    pub trait ParIter {
        type Item;
        fn par_iter(&self) -> std::slice::Iter<'_, Self::Item>;
    }
    impl<T> ParIter for [T] {
        type Item = T;
        fn par_iter(&self) -> std::slice::Iter<'_, T> {
            self.iter()
        }
    }
    impl<T> ParIter for Vec<T> {
        type Item = T;
        fn par_iter(&self) -> std::slice::Iter<'_, T> {
            self.iter()
        }
    }

    /// rayon's `flat_map_iter` == std's `flat_map`.
    pub trait FlatMapIter: Iterator + Sized {
        fn flat_map_iter<U, F>(self, f: F) -> std::iter::FlatMap<Self, U, F>
        where
            U: IntoIterator,
            F: FnMut(Self::Item) -> U,
        {
            self.flat_map(f)
        }
    }
    impl<I: Iterator> FlatMapIter for I {}
}

#[cfg(target_arch = "wasm32")]
pub use seq::{FlatMapIter, IntoParIter, ParIter};
