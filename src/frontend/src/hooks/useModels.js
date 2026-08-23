import { useState, useEffect } from 'react';
import { apiFetch } from '../api';

const KNOWN_MODELS = {
  // Krea 2
  'krea2_turbo_int8_convrot.safetensors': { label: 'Krea 2 Turbo (INT8)', desc: 'Fast, expressive, 8 steps' },
  // FLUX
  'flux1-dev-fp8': { label: 'Flux.1 Dev (FP8)', desc: 'Highest quality, slower' },
  // FLUX 2 (GGUF, multi-GPU)
  'flux2-dev-Q4_K_M.gguf': { label: 'Flux.2 Dev (GGUF Q4)', desc: '32B, multi-GPU, 20 steps' },
  // SDXL-Pony
  'CyberRealistic_Pony_v14.1_FP16.safetensors': { label: 'CyberRealistic Pony', desc: 'Photorealistic + Pony tags' },
  'ponyDiffusionV6XL_v6StartWithThisOne.safetensors': { label: 'Pony Diffusion V6', desc: 'Booru tags, NSFW' },
  'reapony_v90.safetensors': { label: 'Reapony V9', desc: 'Realistic + Pony' },
};

export function useModels(type = 'sdxl') {
  const [modelGroups, setModelGroups] = useState({});
  const [allModels, setAllModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    const fetchModels = async () => {
      try {
        setLoading(true);
        const data = await apiFetch('/api/models/checkpoints');
        if (mounted && data.checkpoints) {
            const mapped = data.checkpoints.map(c => {
                const known = KNOWN_MODELS[c] || {};
                let label = known.label || c.replace('.safetensors', '').replace(/_/g, ' ');
                let desc = known.desc || c;
                return { value: c, label: label, desc: desc };
            });

            // Build the grouped structure
            const groups = {
              "flux": {
                label: '⚡ Flux',
                desc: 'Flux.1 Models',
                models: mapped.filter(m => m.value.toLowerCase().includes('flux') || m.value.toLowerCase().includes('fp8'))
              },
              "sdxl": {
                label: '🎨 SDXL',
                desc: 'Stable Diffusion XL',
                models: mapped.filter(m => !m.value.toLowerCase().includes('flux') && !m.value.toLowerCase().includes('fp8'))
              }
            };

            // Remove empty groups
            for (let [key, group] of Object.entries(groups)) {
               if (group.models.length === 0) {
                  delete groups[key];
               }
            }

            setModelGroups(groups);
            setAllModels(mapped);
        }
      } catch (err) {
        if (mounted) {
           console.error("Failed to load models", err);
           setError(err);
        }
      } finally {
        if (mounted) setLoading(false);
      }
    };
    fetchModels();
    return () => { mounted = false; };
  }, [type]);

  return { modelGroups, allModels, loading, error };
}
