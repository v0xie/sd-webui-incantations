# sd-webui-incantations
This extension implements a number of novel algorithms that aim to enhance image quality, prompt following, and more.

---
### "Perturbed Attention Guidance"
https://arxiv.org/abs/2403.17377  
Alternative to CFG that increases sampling quality.

#### Controls
* **PAG Scale**: Controls the intensity of effect of PAG on the generated image.  

#### Results
Prompt: "a puppy and a kitten on the moon"
- SD 1.5  
![image](./images/xyz_grid-3040-1-a%20puppy%20and%20a%20kitten%20on%20the%20moon.png)

- SD XL  
![image](./images/xyz_grid-3041-1-a%20puppy%20and%20a%20kitten%20on%20the%20moon.jpg)

---
### "Seek for Incantations"
https://arxiv.org/abs/2401.06345  
Generates an image following the prompt, then uses CLIP text/image similarity to add on to the prompt and generate a new image.  

* Original Prompt: cinematic 4K photo of a dog riding a bus and eating cake and wearing headphones  
* Modified Prompt: cinematic 4K photo of a dog riding a bus and eating cake and wearing headphones BREAK - - - - - dog - - bus - - - - - -
![image](./images/xyz_grid-2652-1419902843-cinematic%204K%20photo%20of%20a%20dog%20riding%20a%20bus%20and%20eating%20cake%20and%20wearing%20headphones.png)

---
### "Multi-Concept T2I-Zero"
https://arxiv.org/abs/2310.07419

#### Corrections by Similarities
Reduces the contribution of tokens on far away or conceptually unrelated tokens.

#### Cross-Token Non-Maximum Suppression
Reduces the mixing of features of unrelated concepts.
The implementation of Cross-Token Non-Maximum Suppression is most likely wrong since it's working with the output of the cross-attention modules after attention has been calculated; It produces interesting output despite this.  

Prompt: "A photo of a lion and a grizzly bear and a tiger in the woods"  
![image](./images/xyz_grid-2660-1590472902-A%20photo%20of%20a%20lion%20and%20a%20grizzly%20bear%20and%20a%20tiger%20in%20the%20woods.jpg)  

---
### CADS
https://arxiv.org/abs/2310.17347

[https://github.com/v0xie/sd-webui-cads](https://github.com/v0xie/sd-webui-cads)  

todo...

---
### Semantic Guidance
https://arxiv.org/abs/2301.12247

[https://github.com/v0xie/sd-webui-semantic-guidance](https://github.com/v0xie/sd-webui-semantic-guidance)  

todo...

---

### Issues / Pull Requests are welcome!
---

### Credits
- The authors of the papers for their methods:  

      @misc{yu2024seek,
       title={Seek for Incantations: Towards Accurate Text-to-Image Diffusion Synthesis through Prompt Engineering}, 
       author={Chang Yu and Junran Peng and Xiangyu Zhu and Zhaoxiang Zhang and Qi Tian and Zhen Lei},
       year={2024},
       eprint={2401.06345},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
      }

      @misc{tunanyan2023multiconcept,
       title={Multi-Concept T2I-Zero: Tweaking Only The Text Embeddings and Nothing Else}, 
       author={Hazarapet Tunanyan and Dejia Xu and Shant Navasardyan and Zhangyang Wang and Humphrey Shi},
       year={2023},
       eprint={2310.07419},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
      }

      @misc{ahn2024selfrectifying,
       title={Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance}, 
       author={Donghoon Ahn and Hyoungwon Cho and Jaewon Min and Wooseok Jang and Jungwoo Kim and SeonHwa Kim and Hyun Hee Park and Kyong Hwan Jin and Seungryong Kim},
       year={2024},
       eprint={2403.17377},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
      }

      @misc{sadat2023cads,
       title={CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling},
       author={Seyedmorteza Sadat and Jakob Buhmann and Derek Bradely and Otmar Hilliges and Romann M. Weber},
       year={2023},
       eprint={2310.17347},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
      }

      @misc{brack2023sega,
       title={SEGA: Instructing Text-to-Image Models using Semantic Guidance}, 
       author={Manuel Brack and Felix Friedrich and Dominik Hintersdorf and Lukas Struppek and Patrick Schramowski and Kristian Kersting},
       year={2023},
       eprint={2301.12247},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
      }

- Hard Prompts Made Easy (https://github.com/YuxinWenRick/hard-prompts-made-easy)

- @udon-universe's extension templates (https://github.com/udon-universe/stable-diffusion-webui-extension-templates)
---
