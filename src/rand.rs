use crate::*;

const N: usize = 624;
const M: usize = 397;

/// Random number generator using the Mersenne Twister algorithm.
pub struct Rng {
    state: [u32; N],
    index: u32,
}

impl Rng {
    /// Initializes the random number seed with a u32 value.
    pub fn new(seed: u32) -> Rng {
        let mut mt = Rng {
            state: [0; N],
            index: N as u32,
        };
        mt.state[0] = seed;
        for i in 1..N {
            mt.state[i] = (1812433253_u32.wrapping_mul(mt.state[i - 1] ^ (mt.state[i - 1] >> 30)) + i as u32) & 0xffffffff;
        }
        mt
    }

    /// Generates a uniformly distributed random number in the range `[0, 1)`.
    pub fn gen(&mut self) -> X64 {
        if self.index >= N as u32 {
            self.twist();
        }

        let mut y = self.state[self.index as usize];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680_u32;
        y ^= (y << 15) & 0xefc60000_u32;
        y ^= y >> 18;

        self.index += 1;
        X64::from_bits(y as i64)
    }

    fn twist(&mut self) {
        for i in 0..N {
            let x = (self.state[i] & 0x80000000_u32) + (self.state[(i + 1) % N] & 0x7fffffff_u32);
            let mut x_a = x >> 1;
            if x % 2 != 0 {
                x_a ^= 0x9908b0df_u32;
            }
            self.state[i] = self.state[(i + M) % N] ^ x_a;
        }
        self.index = 0;
    }
}
