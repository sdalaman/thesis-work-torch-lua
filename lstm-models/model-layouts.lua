BiLangModelLSTMScAvg
---------------------------------

nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> output]
  (1): nn.Sequencer @ nn.Recursor @ nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.LookupTableMaskZero
    (2): nn.ConcatTable {
      input
        |`-> (1): nn.Identity
         `-> (2): nn.Identity
         ... -> output
    }
    (3): nn.ParallelTable {
      input
        |`-> (1): nn.Sequential {
        |      [input -> (1) -> (2) -> (3) -> output]
        |      (1): nn.Linear(256 -> 512)
        |      (2): nn.Sequential {
        |        [input -> (1) -> (2) -> output]
        |        (1): nn.LSTM(512 -> 512)
        |        (2): nn.NormStabilizer
        |      }
        |      (3): nn.Linear(512 -> 256)
        |    }
         `-> (2): nn.Identity
         ... -> output
    }
  }
  (2): nn.FlattenTable
  (3): nn.CAddTable
  (4): nn.MulConstant
}




BiLangModelBiLSTMScAvg
-----------------------------

nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nn.ParallelTable {
    input
      |`-> (1): nn.Sequential {
      |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |      (1): nn.Sequencer @ nn.Recursor @ nn.Sequential {
      |        [input -> (1) -> (2) -> (3) -> output]
      |        (1): nn.LookupTableMaskZero
      |        (2): nn.ConcatTable {
      |          input
      |            |`-> (1): nn.Identity
      |             `-> (2): nn.Identity
      |             ... -> output
      |        }
      |        (3): nn.ParallelTable {
      |          input
      |            |`-> (1): nn.Sequential {
      |            |      [input -> (1) -> (2) -> (3) -> output]
      |            |      (1): nn.Linear(64 -> 512)
      |            |      (2): nn.Sequential {
      |            |        [input -> (1) -> (2) -> output]
      |            |        (1): nn.LSTM(512 -> 512)
      |            |        (2): nn.NormStabilizer
      |            |      }
      |            |      (3): nn.Linear(512 -> 64)
      |            |    }
      |             `-> (2): nn.Identity
      |             ... -> output
      |        }
      |      }
      |      (2): nn.FlattenTable
      |      (3): nn.CAddTable
      |      (4): nn.MulConstant
      |    }
       `-> (2): nn.Sequential {
             [input -> (1) -> (2) -> (3) -> (4) -> output]
             (1): nn.Sequencer @ nn.Recursor @ nn.Sequential {
               [input -> (1) -> (2) -> (3) -> output]
               (1): nn.LookupTableMaskZero
               (2): nn.ConcatTable {
                 input
                   |`-> (1): nn.Identity
                    `-> (2): nn.Identity
                    ... -> output
               }
               (3): nn.ParallelTable {
                 input
                   |`-> (1): nn.Sequential {
                   |      [input -> (1) -> (2) -> (3) -> output]
                   |      (1): nn.Linear(64 -> 512)
                   |      (2): nn.Sequential {
                   |        [input -> (1) -> (2) -> output]
                   |        (1): nn.LSTM(512 -> 512)
                   |        (2): nn.NormStabilizer
                   |      }
                   |      (3): nn.Linear(512 -> 64)
                   |    }
                    `-> (2): nn.Identity
                    ... -> output
               }
             }
             (2): nn.FlattenTable
             (3): nn.CAddTable
             (4): nn.MulConstant
           }
       ... -> output
  }
  (2): nn.JoinTable
}
