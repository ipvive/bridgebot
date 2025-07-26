```asciiart
     deal object
         |
 {comprehend shim}  <-------- dims list
         |
=========================================================================
absolute answers vector                        action index
         |                                          |
     {FSA(n)}                                {action-to-grid}
         |                                          |
absolute answers vector - input        one-hot (5x7 + 1x3 + 4x13)
         |                                          |
  {preprocessor - hk.Embed}             {black and white image embed}
         |                                          |
  inputs with trainable pos           1 white pixel with 2d fourier pos
         |                 \                       /
         |              {encoder}             {encoder}
         |                   \                   /
         |                    {multimodal encoder}
         |                           /
         |               latent array
         |                    |
         |           {latent transformer}
         |                    |
         |               latent array
         |                   /
         |     --------------
         |    /
   {EmbeddingDecoder}         {FSA(n+1)}
         |                       |
      logits          absolute answers vector - true output
         |          /
{cross-entropy loss}
```
