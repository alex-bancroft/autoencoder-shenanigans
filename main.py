from tqdm import tqdm
from functools import partial
from sae_lens import ActivationsStore, SAE, HookedSAETransformer

max_tokens = 50
models = {
    "gpt2-small": {
        "release": "gpt2-small-res-jb",
        "sae": "blocks.7.hook_resid_pre",
    },
    "gemma-2-2b": {
        "release": "gemma-scope-2b-pt-mlp-canonical",
        # TODO: this is wrong
        "sae": "layer_24/width_16k/canonical",
    }
}

model_name = "gpt2-small"

# Define Stuff
device = "cuda"
model = HookedSAETransformer.from_pretrained(model_name, device = device)
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = models[model_name]["release"], # <- Release name 
    sae_id = models[model_name]["sae"], # <- SAE id (not always a hook point!)
    device = device
)

# a convenient way to instantiate an activation store is to use the from_sae method
activation_store = ActivationsStore.from_sae(
    model=model,
    sae=sae,
    streaming=True,
    # fairly conservative parameters here so can use same for larger
    # models without running out of memory.
    store_batch_size_prompts=8,
    train_batch_size_tokens=4096,
    n_batches_in_buffer=32,
    device=device,
)

def find_max_activation(model, sae, activation_store, feature_idx, num_batches=100):
    '''
    Find the maximum activation for a given feature index. This is useful for 
    calibrating the right amount of the feature to add.
    '''
    max_activation = 0.0

    pbar = tqdm(range(num_batches))
    for _ in pbar:
        tokens = activation_store.get_batch_tokens()
        
        _, cache = model.run_with_cache(
            tokens, 
            stop_at_layer=sae.cfg.hook_layer + 1, 
            names_filter=[sae.cfg.hook_name]
        )
        sae_in = cache[sae.cfg.hook_name]
        feature_acts = sae.encode(sae_in).squeeze()

        feature_acts = feature_acts.flatten(0, 1)
        batch_max_activation = feature_acts[:, feature_idx].max().item()
        max_activation = max(max_activation, batch_max_activation)
        
        pbar.set_description(f"Max activation: {max_activation:.4f}")

    return max_activation

def steering(activations, hook, steering_strength=1.0, steering_vector=None, max_act=1.0):
    # Note if the feature fires anyway, we'd be adding to that here.
    return activations + max_act * steering_strength * steering_vector

def generate_with_steering(model, sae, prompt, steering_feature, max_act, steering_strength=1.0, max_new_tokens=max_tokens):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    
    steering_vector = sae.W_dec[steering_feature].to(model.cfg.device)
    
    steering_hook = partial(
        steering,
        steering_vector=steering_vector,
        steering_strength=steering_strength,
        max_act=max_act
    )
    
    # standard transformerlens syntax for a hook context for generation
    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos = False if device == "mps" else True,
            prepend_bos = sae.cfg.prepend_bos,
        )
    
    return model.tokenizer.decode(output[0])

# Choose a feature to steer
features = {
    "gpt2-small": {
        "animals": 12952,
        "light": 17519,
        "finances": 17522,
    },
    "gemma-2-2b": {
        "pollutants": 1500,
    }
}
steering_feature = steering_feature = features[model_name]["finances"]  # Choose a feature to steer towards

# Find the maximum activation for this feature
max_act = find_max_activation(model, sae, activation_store, steering_feature)
print(f"Maximum activation for feature {steering_feature}: {max_act:.4f}")

# note we could also get the max activation from Neuronpedia (https://www.neuronpedia.org/api-doc#tag/lookup/GET/api/feature/{modelId}/{layer}/{index})

# Generate text without steering for comparison
prompt = "The most important meal of the day is"
normal_text = model.generate(
    prompt,
    max_new_tokens=max_tokens, 
    stop_at_eos = False if device == "mps" else True,
    prepend_bos = sae.cfg.prepend_bos,
)

print("\nNormal text (without steering):")
print(normal_text)

# Generate text with steering
steered_text = generate_with_steering(model, sae, prompt, steering_feature, max_act, steering_strength=2.0)
print("Steered text:")
print(steered_text)