#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, Address, FixedBytes};
//use byteorder::{BigEndian, ByteOrder, LittleEndian};
//use byteorder::{LittleEndian, WriteBytesExt,BigEndian};
use byteorder::{LittleEndian, WriteBytesExt};
use console::Term;
use fs4::FileExt;
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use separator::Separatable;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};
use tiny_keccak::{Hasher, Keccak};

mod reward;
pub use reward::Reward;

// workset size (tweak this!)
const WORK_SIZE: u32 = 0x4000000; // max. 0x15400000 to abs. max 0xffffffff

const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;
const CONTROL_CHARACTER: u8 = 0xff;
const MAX_INCREMENTER: u64 = 0xffffffffffff;

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

/// Requires three hex-encoded arguments: the address of the contract that will
/// be calling CREATE2, the address of the caller of said contract *(assuming
/// the contract calling CREATE2 has frontrunning protection in place - if not
/// applicable to your use-case you can set it to the null address)*, and the
/// keccak-256 hash of the bytecode that is provided by the contract calling
/// CREATE2 that will be used to initialize the new contract. An additional set
/// of three optional values may be provided: a device to target for OpenCL GPU
/// search, a threshold for leading zeroes to search for, and a threshold for
/// total zeroes to search for.
pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
    pub gpu_device: u8,
    pub leading_zeroes_threshold: u8,
    pub total_zeroes_threshold: u8,
}

/// Validate the provided arguments and construct the Config struct.
impl Config {
    pub fn new(mut args: std::env::Args) -> Result<Self, &'static str> {
        // get args, skipping first arg (program name)
        args.next();

        let Some(factory_address_string) = args.next() else {
            return Err("didn't get a factory_address argument");
        };
        let Some(calling_address_string) = args.next() else {
            return Err("didn't get a calling_address argument");
        };
        let Some(init_code_hash_string) = args.next() else {
            return Err("didn't get an init_code_hash argument");
        };

        let gpu_device_string = match args.next() {
            Some(arg) => arg,
            None => String::from("255"), // indicates that CPU will be used.
        };
        let leading_zeroes_threshold_string = match args.next() {
            Some(arg) => arg,
            None => String::from("3"),
        };
        let total_zeroes_threshold_string = match args.next() {
            Some(arg) => arg,
            None => String::from("5"),
        };

        // convert main arguments from hex string to vector of bytes
        let Ok(factory_address_vec) = hex::decode(factory_address_string) else {
            return Err("could not decode factory address argument");
        };
        let Ok(calling_address_vec) = hex::decode(calling_address_string) else {
            return Err("could not decode calling address argument");
        };
        let Ok(init_code_hash_vec) = hex::decode(init_code_hash_string) else {
            return Err("could not decode initialization code hash argument");
        };

        // convert from vector to fixed array
        let Ok(factory_address) = factory_address_vec.try_into() else {
            return Err("invalid length for factory address argument");
        };
        let Ok(calling_address) = calling_address_vec.try_into() else {
            return Err("invalid length for calling address argument");
        };
        let Ok(init_code_hash) = init_code_hash_vec.try_into() else {
            return Err("invalid length for initialization code hash argument");
        };

        // convert gpu arguments to u8 values
        let Ok(gpu_device) = gpu_device_string.parse::<u8>() else {
            return Err("invalid gpu device value");
        };
        let Ok(leading_zeroes_threshold) = leading_zeroes_threshold_string.parse::<u8>() else {
            return Err("invalid leading zeroes threshold value supplied");
        };
        let Ok(total_zeroes_threshold) = total_zeroes_threshold_string.parse::<u8>() else {
            return Err("invalid total zeroes threshold value supplied");
        };

        if leading_zeroes_threshold > 20 {
            return Err("invalid value for leading zeroes threshold argument. (valid: 0..=20)");
        }
        if total_zeroes_threshold > 20 && total_zeroes_threshold != 255 {
            return Err("invalid value for total zeroes threshold argument. (valid: 0..=20 | 255)");
        }

        Ok(Self {
            factory_address,
            calling_address,
            init_code_hash,
            gpu_device,
            leading_zeroes_threshold,
            total_zeroes_threshold,
        })
    }
}

/// Given a Config object with a factory address, a caller address, a keccak-256
/// hash of the contract initialization code, and a device ID, search for salts
/// using OpenCL that will enable the factory contract to deploy a contract to a
/// gas-efficient address via CREATE2. This method also takes threshold values
/// for both leading zero bytes and total zero bytes - any address that does not
/// meet or exceed the threshold will not be returned. Default threshold values
/// are three leading zeroes or five total zeroes.
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 4-byte segment (to prevent collisions with other runs)
///   - a 4-byte segment unique to each work group running in parallel
///   - a 4-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. approximate rarity) of the
/// resultant address.
///
/// This method is still highly experimental and could almost certainly use
/// further optimization - contributions are more than welcome!

pub fn gpu(config: Config) -> ocl::Result<()> {
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        config.gpu_device
    );

    // (create if necessary) and open a file where found salts will be written
    let _file = output_file(); // Use `_file` if not used to suppress warning

    // create object for computing rewards (relative rarity) for a given address
    let _rewards = Reward::new(); // Use `_rewards` if not used to suppress warning

    // track how many addresses have been found and information about them
    let mut _found: u64 = 0; // Use `_found` if not used to suppress warning
    let mut _found_list: Vec<String> = vec![]; // Use `_found_list` if not used to suppress warning

    // set up a controller for terminal output
    let _term = Term::stdout(); // Use `_term` if not used to suppress warning

    // set up a platform to use
    let platform = Platform::new(ocl::core::default_platform()?);

    // set up the device to use
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;

    // set up the context to use
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // set up the program to use
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .build(&context)?;

    // set up the queue to use
    let queue = Queue::new(&context, device, None)?;

    // set up the "proqueue" (or amalgamation of various elements) to use
    let ocl_pq = ProQue::new(context, queue, program, Some(WORK_SIZE));

    // create a random number generator
    let mut rng = thread_rng();

    // Initialize highest score
    let mut highest_score = 0;

    // Begin searching for addresses
    loop {
        // Generate a random salt
        let salt = FixedBytes::<4>::random();

        // Build the message buffer
        let message_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;

        // Initialize the nonce
        let nonce_init = rng.gen::<u32>();
        let mut nonce = [nonce_init];

        // Build the nonce buffer
        let nonce_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&nonce)
            .build()?;

        // Build the results buffer
        let results_size = 6 * WORK_SIZE as usize;
        let mut results: Vec<u32> = vec![0; results_size];
        let results_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(results_size)
            .build()?;

        // Build the kernel
let kern = ocl_pq
.kernel_builder("hashMessage")
.arg_named("message", &message_buffer)
.arg_named("nonce", &nonce_buffer)
.arg_named("results", &results_buffer)
.build()?;

// Inner loop to keep executing the kernel with updated nonce
loop {
// Enqueue the kernel
unsafe { kern.enq()? };

// Read the results
results_buffer.read(&mut results).enq()?;

// Process the results
for i in 0..WORK_SIZE as usize {
    let idx = i * 6;
    let score = results[idx];
    if score > highest_score {
        highest_score = score;
        // Extract the address bytes
        let mut address_bytes = [0u8; 20];
        for j in 0..5 {
            let val = results[idx + 1 + j];
            address_bytes[4 * j] = (val >> 24) as u8;
            address_bytes[4 * j + 1] = (val >> 16) as u8;
            address_bytes[4 * j + 2] = (val >> 8) as u8;
            address_bytes[4 * j + 3] = val as u8;
        }
        // Convert address to hex string
        let address_hex = hex::encode(address_bytes);

        // Corrected: Use the correct order of nonce components
        let nonce_uint32_t = [i as u32, nonce[0]];
        let mut nonce_bytes = [0u8; 8];
        {
            let mut cursor = std::io::Cursor::new(&mut nonce_bytes[..]);
            cursor.write_u32::<LittleEndian>(nonce_uint32_t[0])?; // i as u32
            cursor.write_u32::<LittleEndian>(nonce_uint32_t[1])?; // nonce[0]
        }

        // Construct full_salt: calling_address + salt + nonce_bytes
        let full_salt = [
            &config.calling_address[..], // 20 bytes
            &salt[..],                   // 4 bytes
            &nonce_bytes[..],            // 8 bytes
        ]
        .concat();

        // Verify the address
        let mut data = Vec::with_capacity(1 + 20 + 32 + 32);
        data.push(0xffu8); // Control character
        data.extend_from_slice(&config.factory_address); // Factory address
        data.extend_from_slice(&full_salt); // Full salt (includes calling address)
        data.extend_from_slice(&config.init_code_hash); // Init code hash

        // Hash data using Keccak-256
        let mut hasher = Keccak::v256();
        hasher.update(&data);
        let mut address_hash = [0u8; 32];
        hasher.finalize(&mut address_hash);

        // Extract the address
        let address_bytes_from_hash = &address_hash[12..]; // Last 20 bytes
        let address_hex_from_hash = hex::encode(address_bytes_from_hash);

        // Compare the addresses
        if address_hex_from_hash != address_hex {
            println!(
                "Address mismatch! Computed: {}, Expected: {}",
                address_hex_from_hash, address_hex
            );
        } else {
            println!("Address verified successfully.");
        }

        // Output the result
        let output = format!(
            "0x{} => {} => {}",
            hex::encode(full_salt),
            address_hex,
            highest_score,
        );
        println!("{}", output);
    }
}

// Increment the nonce
nonce[0] += 1;

// Update the nonce buffer
nonce_buffer.write(&nonce[..]).enq()?; // Use &nonce[..] here
        }
    }
}


#[track_caller]
fn output_file() -> File {
    OpenOptions::new()
        .append(true)
        .create(true)
        .read(true)
        .open("efficient_addresses.txt")
        .expect("Could not create or open `efficient_addresses.txt` file.")
}

/// Creates the OpenCL kernel source code by populating the template with the
/// values from the Config object.
fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

    let factory = config.factory_address.iter();
    let caller = config.calling_address.iter();
    let hash = config.init_code_hash.iter();
    let hash = hash.enumerate().map(|(i, x)| (i + 52, x));
    for (i, x) in factory.chain(caller).enumerate().chain(hash) {
        writeln!(src, "#define S_{} {}u", i + 1, x).unwrap();
    }
    let lz = config.leading_zeroes_threshold;
    writeln!(src, "#define LEADING_ZEROES {lz}").unwrap();
    let tz = config.total_zeroes_threshold;
    writeln!(src, "#define TOTAL_ZEROES {tz}").unwrap();

    src.push_str(KERNEL_SRC);

    src
}
