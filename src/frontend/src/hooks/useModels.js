import { useState, useEffect } from 'react';
import { apiFetch } from '../api';

const KNOWN_MODELS = {
  // FLUX
  'flux1-dev-fp8': { label: 'Flux.1 Dev (FP8)', desc: 'Highest quality, slower' },
  // SDXL
  'CyberRealistic_Pony_v14.1_FP16.safetensors': { label: 'CyberRealistic Pony', desc: 'Photorealistic + Pony tags' },
  'dreamshaperXL_lightningDPMSDE.safetensors': { label: 'Dreamshaper Lightning', desc: 'Fast, artistic' },
  'illustriousRealismBy_v10VAE.safetensors': { label: 'Illustrious Realism', desc: 'Detailed realistic' },
  'juggernautXL_ragnarok.safetensors': { label: 'Juggernaut XL', desc: 'All-rounder' },
  'novaAnimeXL_ilV150.safetensors': { label: 'Nova Anime XL', desc: 'Anime style' },
  'ponyDiffusionV6XL_v6StartWithThisOne.safetensors': { label: 'Pony Diffusion V6', desc: 'Booru tags, NSFW' },
  'reapony_v90.safetensors': { label: 'Reapony V9', desc: 'Realistic + Pony' },
  'ultraRealisticByStable_v20FP16.safetensors': { label: 'Ultra Realistic', desc: 'Hyperrealistic' },
  'waiIllustriousSDXL_v160.safetensors': { label: 'Wai Illustrious', desc: 'Anime + 2.5D' },
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
