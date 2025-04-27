use core::fmt::Debug;
use core::ops::{Div, Mul, Rem};
use hybrid_array::{
    typenum::{Gcd, Gcf, Prod, Quot, Unsigned, U0, U256, U32, U8},
    Array,
};
use num_traits::One;

use super::algebra::{Elem, Field, NttPolynomial, NttVector, Polynomial, Vector};
use super::util::{Flatten, Truncate, Unflatten};

/// An array length with other useful properties
pub trait ArraySize: hybrid_array::ArraySize + PartialEq + Debug {}

impl<T> ArraySize for T where T: hybrid_array::ArraySize + PartialEq + Debug {}

/// An integer that can describe encoded polynomials.
pub trait EncodingSize: ArraySize {
    type EncodedPolynomialSize: ArraySize;
    type ValueStep: ArraySize;
    type ByteStep: ArraySize;
}

type EncodingUnit<D> = Quot<Prod<D, U8>, Gcf<D, U8>>;

pub type EncodedPolynomialSize<D> = <D as EncodingSize>::EncodedPolynomialSize;
pub type EncodedPolynomial<D> = Array<u8, EncodedPolynomialSize<D>>;

impl<D> EncodingSize for D
where
    D: ArraySize + Mul<U8> + Gcd<U8> + Mul<U32>,
    Prod<D, U32>: ArraySize,
    Prod<D, U8>: Div<Gcf<D, U8>>,
    EncodingUnit<D>: Div<D> + Div<U8>,
    Quot<EncodingUnit<D>, D>: ArraySize,
    Quot<EncodingUnit<D>, U8>: ArraySize,
{
    type EncodedPolynomialSize = Prod<D, U32>;
    type ValueStep = Quot<EncodingUnit<D>, D>;
    type ByteStep = Quot<EncodingUnit<D>, U8>;
}

type DecodedValue<F> = Array<Elem<F>, U256>;

/// An integer that can describe encoded vectors.
pub trait VectorEncodingSize<K>: EncodingSize
where
    K: ArraySize,
{
    type EncodedVectorSize: ArraySize;

    fn flatten(polys: Array<EncodedPolynomial<Self>, K>) -> EncodedVector<Self, K>;
    fn unflatten(vec: &EncodedVector<Self, K>) -> Array<&EncodedPolynomial<Self>, K>;
}

pub type EncodedVectorSize<D, K> = <D as VectorEncodingSize<K>>::EncodedVectorSize;
pub type EncodedVector<D, K> = Array<u8, EncodedVectorSize<D, K>>;

impl<D, K> VectorEncodingSize<K> for D
where
    D: EncodingSize,
    K: ArraySize,
    D::EncodedPolynomialSize: Mul<K>,
    Prod<D::EncodedPolynomialSize, K>:
        ArraySize + Div<K, Output = D::EncodedPolynomialSize> + Rem<K, Output = U0>,
{
    type EncodedVectorSize = Prod<D::EncodedPolynomialSize, K>;

    fn flatten(polys: Array<EncodedPolynomial<Self>, K>) -> EncodedVector<Self, K> {
        polys.flatten()
    }

    fn unflatten(vec: &EncodedVector<Self, K>) -> Array<&EncodedPolynomial<Self>, K> {
        vec.unflatten()
    }
}

// FIPS 203: Algorithm 4 ByteEncode_d
// FIPS 204: Algorithm 16 SimpleBitPack
fn byte_encode<F: Field, D: EncodingSize>(vals: &DecodedValue<F>) -> EncodedPolynomial<D> {
    let val_step = D::ValueStep::USIZE;
    let byte_step = D::ByteStep::USIZE;

    let mut bytes = EncodedPolynomial::<D>::default();

    let vc = vals.chunks(val_step);
    let bc = bytes.chunks_mut(byte_step);
    for (v, b) in vc.zip(bc) {
        let mut x = 0u128;
        for (j, vj) in v.iter().enumerate() {
            let vj: u128 = vj.0.into();
            x |= vj << (D::USIZE * j);
        }

        let xb = x.to_le_bytes();
        b.copy_from_slice(&xb[..byte_step]);
    }

    bytes
}

// FIPS 203: Algorithm 5 ByteDecode_d(F)
// FIPS 204: Algorithm 18 SimpleBitUnpack
fn byte_decode<F: Field, D: EncodingSize>(bytes: &EncodedPolynomial<D>) -> DecodedValue<F> {
    let val_step = D::ValueStep::USIZE;
    let byte_step = D::ByteStep::USIZE;
    let mask = (F::Int::one() << D::USIZE) - F::Int::one();

    let mut vals = DecodedValue::default();

    let vc = vals.chunks_mut(val_step);
    let bc = bytes.chunks(byte_step);
    for (v, b) in vc.zip(bc) {
        let mut xb = [0u8; 16];
        xb[..byte_step].copy_from_slice(b);

        let x = u128::from_le_bytes(xb);
        for (j, vj) in v.iter_mut().enumerate() {
            let val = F::Int::truncate(x >> (D::USIZE * j));
            vj.0 = val & mask;

            // Special case for FIPS 203
            if D::USIZE == 12 {
                vj.0 = vj.0 % F::Q;
            }
        }
    }

    vals
}

pub trait Encode<D: EncodingSize> {
    type EncodedSize: ArraySize;
    fn encode(&self) -> Array<u8, Self::EncodedSize>;
    fn decode(enc: &Array<u8, Self::EncodedSize>) -> Self;
}

impl<F: Field, D: EncodingSize> Encode<D> for Polynomial<F> {
    type EncodedSize = D::EncodedPolynomialSize;

    fn encode(&self) -> Array<u8, Self::EncodedSize> {
        byte_encode::<F, D>(&self.0)
    }

    fn decode(enc: &Array<u8, Self::EncodedSize>) -> Self {
        Self(byte_decode::<F, D>(enc))
    }
}

impl<F, D, K> Encode<D> for Vector<F, K>
where
    F: Field,
    K: ArraySize,
    D: VectorEncodingSize<K>,
{
    type EncodedSize = D::EncodedVectorSize;

    fn encode(&self) -> Array<u8, Self::EncodedSize> {
        let polys = self.0.iter().map(|x| Encode::<D>::encode(x)).collect();
        <D as VectorEncodingSize<K>>::flatten(polys)
    }

    fn decode(enc: &Array<u8, Self::EncodedSize>) -> Self {
        let unfold = <D as VectorEncodingSize<K>>::unflatten(enc);
        Self(
            unfold
                .iter()
                .map(|&x| <Polynomial<F> as Encode<D>>::decode(x))
                .collect(),
        )
    }
}

impl<F: Field, D: EncodingSize> Encode<D> for NttPolynomial<F> {
    type EncodedSize = D::EncodedPolynomialSize;

    fn encode(&self) -> Array<u8, Self::EncodedSize> {
        byte_encode::<F, D>(&self.0)
    }

    fn decode(enc: &Array<u8, Self::EncodedSize>) -> Self {
        Self(byte_decode::<F, D>(enc))
    }
}

impl<F, D, K> Encode<D> for NttVector<F, K>
where
    F: Field,
    D: VectorEncodingSize<K>,
    K: ArraySize,
{
    type EncodedSize = D::EncodedVectorSize;

    fn encode(&self) -> Array<u8, Self::EncodedSize> {
        let polys = self.0.iter().map(|x| Encode::<D>::encode(x)).collect();
        <D as VectorEncodingSize<K>>::flatten(polys)
    }

    fn decode(enc: &Array<u8, Self::EncodedSize>) -> Self {
        let unfold = <D as VectorEncodingSize<K>>::unflatten(enc);
        Self(
            unfold
                .iter()
                .map(|&x| <NttPolynomial<F> as Encode<D>>::decode(x))
                .collect(),
        )
    }
}

// #[cfg(test)]
// pub(crate) mod test {
//     use super::*;
//     use core::fmt::Debug;
//     use core::ops::Rem;
//     use hybrid_array::typenum::{
//         marker_traits::Zero, operator_aliases::Mod, U1, U10, U11, U12, U2, U3, U4, U5, U6, U8,
//     };
//     use rand::Rng;

//     use crate::param::EncodedPolynomialVector;

//     // A helper trait to construct larger arrays by repeating smaller ones
//     trait Repeat<T: Clone, D: ArraySize> {
//         fn repeat(&self) -> Array<T, D>;
//     }

//     impl<T, N, D> Repeat<T, D> for Array<T, N>
//     where
//         N: ArraySize,
//         T: Clone,
//         D: ArraySize + Rem<N>,
//         Mod<D, N>: Zero,
//     {
//         #[allow(clippy::integer_division_remainder_used)]
//         fn repeat(&self) -> Array<T, D> {
//             Array::from_fn(|i| self[i % N::USIZE].clone())
//         }
//     }

//     #[allow(clippy::integer_division_remainder_used)]
//     fn byte_codec_test<D>(decoded: &DecodedValue, encoded: &EncodedPolynomial<D>)
//     where
//         D: EncodingSize,
//     {
//         // Test known answer
//         let actual_encoded = byte_encode::<D>(decoded);
//         assert_eq!(&actual_encoded, encoded);

//         let actual_decoded = byte_decode::<D>(encoded);
//         assert_eq!(&actual_decoded, decoded);

//         // Test random decode/encode and encode/decode round trips
//         let mut rng = rand::thread_rng();
//         let mut decoded: Array<Integer, U256> = Array::default();
//         rng.fill(decoded.as_mut_slice());
//         let m = match D::USIZE {
//             12 => FieldElement::Q,
//             d => (1 as Integer) << d,
//         };
//         let decoded = decoded.iter().map(|x| FieldElement(x % m)).collect();

//         let actual_encoded = byte_encode::<D>(&decoded);
//         let actual_decoded = byte_decode::<D>(&actual_encoded);
//         assert_eq!(actual_decoded, decoded);

//         let actual_reencoded = byte_encode::<D>(&decoded);
//         assert_eq!(actual_reencoded, actual_encoded);
//     }

//     #[test]
//     fn byte_codec() {
//         // The 1-bit can only represent decoded values equal to 0 or 1.
//         let decoded: DecodedValue = Array::<_, U2>([FieldElement(0), FieldElement(1)]).repeat();
//         let encoded: EncodedPolynomial<U1> = Array([0xaa; 32]);
//         byte_codec_test::<U1>(&decoded, &encoded);

//         // For other codec widths, we use a standard sequence
//         let decoded: DecodedValue = Array::<_, U8>([
//             FieldElement(0),
//             FieldElement(1),
//             FieldElement(2),
//             FieldElement(3),
//             FieldElement(4),
//             FieldElement(5),
//             FieldElement(6),
//             FieldElement(7),
//         ])
//         .repeat();

//         let encoded: EncodedPolynomial<U4> = Array::<_, U4>([0x10, 0x32, 0x54, 0x76]).repeat();
//         byte_codec_test::<U4>(&decoded, &encoded);

//         let encoded: EncodedPolynomial<U5> =
//             Array::<_, U5>([0x20, 0x88, 0x41, 0x8a, 0x39]).repeat();
//         byte_codec_test::<U5>(&decoded, &encoded);

//         let encoded: EncodedPolynomial<U6> =
//             Array::<_, U6>([0x40, 0x20, 0x0c, 0x44, 0x61, 0x1c]).repeat();
//         byte_codec_test::<U6>(&decoded, &encoded);

//         let encoded: EncodedPolynomial<U10> =
//             Array::<_, U10>([0x00, 0x04, 0x20, 0xc0, 0x00, 0x04, 0x14, 0x60, 0xc0, 0x01]).repeat();
//         byte_codec_test::<U10>(&decoded, &encoded);

//         let encoded: EncodedPolynomial<U11> = Array::<_, U11>([
//             0x00, 0x08, 0x80, 0x00, 0x06, 0x40, 0x80, 0x02, 0x18, 0xe0, 0x00,
//         ])
//         .repeat();
//         byte_codec_test::<U11>(&decoded, &encoded);

//         let encoded: EncodedPolynomial<U12> = Array::<_, U12>([
//             0x00, 0x10, 0x00, 0x02, 0x30, 0x00, 0x04, 0x50, 0x00, 0x06, 0x70, 0x00,
//         ])
//         .repeat();
//         byte_codec_test::<U12>(&decoded, &encoded);
//     }

//     #[allow(clippy::integer_division_remainder_used)]
//     #[test]
//     fn byte_codec_12_mod() {
//         // DecodeBytes_12 is required to reduce mod q
//         let encoded: EncodedPolynomial<U12> = Array([0xff; 384]);
//         let decoded: DecodedValue = Array([FieldElement(0xfff % FieldElement::Q); 256]);

//         let actual_decoded = byte_decode::<U12>(&encoded);
//         assert_eq!(actual_decoded, decoded);
//     }

//     fn vector_codec_known_answer_test<D, T>(decoded: &T, encoded: &Array<u8, T::EncodedSize>)
//     where
//         D: EncodingSize,
//         T: Encode<D> + PartialEq + Debug,
//     {
//         let actual_encoded = decoded.encode();
//         assert_eq!(&actual_encoded, encoded);

//         let actual_decoded: T = Encode::decode(encoded);
//         assert_eq!(&actual_decoded, decoded);
//     }

//     #[test]
//     fn vector_codec() {
//         let poly = Polynomial(
//             Array::<_, U8>([
//                 FieldElement(0),
//                 FieldElement(1),
//                 FieldElement(2),
//                 FieldElement(3),
//                 FieldElement(4),
//                 FieldElement(5),
//                 FieldElement(6),
//                 FieldElement(7),
//             ])
//             .repeat(),
//         );

//         // The required vector sizes are 2, 3, and 4.
//         let decoded: PolynomialVector<U2> = PolynomialVector(Array([poly, poly]));
//         let encoded: EncodedPolynomialVector<U5, U2> =
//             Array::<_, U5>([0x20, 0x88, 0x41, 0x8a, 0x39]).repeat();
//         vector_codec_known_answer_test::<U5, PolynomialVector<U2>>(&decoded, &encoded);

//         let decoded: PolynomialVector<U3> = PolynomialVector(Array([poly, poly, poly]));
//         let encoded: EncodedPolynomialVector<U5, U3> =
//             Array::<_, U5>([0x20, 0x88, 0x41, 0x8a, 0x39]).repeat();
//         vector_codec_known_answer_test::<U5, PolynomialVector<U3>>(&decoded, &encoded);

//         let decoded: PolynomialVector<U4> = PolynomialVector(Array([poly, poly, poly, poly]));
//         let encoded: EncodedPolynomialVector<U5, U4> =
//             Array::<_, U5>([0x20, 0x88, 0x41, 0x8a, 0x39]).repeat();
//         vector_codec_known_answer_test::<U5, PolynomialVector<U4>>(&decoded, &encoded);
//     }
// }
