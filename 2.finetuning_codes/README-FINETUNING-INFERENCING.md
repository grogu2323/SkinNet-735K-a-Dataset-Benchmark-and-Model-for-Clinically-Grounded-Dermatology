# MedGemma Fine-Tuning

![MedGemma fine-tuning and inference pipeline](Finetuning_inferencing.png)

Fine-tuning code for MedGemma-4B and MedGemma-27B, used to generate structured clinical descriptors for SkinNet-735K. 
Each case is trained as a single **15-turn conversation**: 
  - one question per clinical attribute,
  - with the image(s) at turn 1 and every earlier question and answer kept in the history for later turns.
