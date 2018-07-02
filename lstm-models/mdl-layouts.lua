####################################################################
----BiLangModelLSTM----
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nn.Sequencer @ nn.Recursor @ nn.Sequential {
    [input -> (1) -> (2) -> (3) -> (4) -> output]
    (1): nn.LookupTableMaskZero
    (2): nn.Linear(128 -> 512)
    (3): nn.LSTM(512 -> 512)
    (4): nn.NormStabilizer
  }
  (2): nn.SelectTable(-1)
}
####################################################################
----BiLangModelLSTMAvg----
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nn.Sequencer @ nn.Recursor @ nn.Sequential {
    [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
    (1): nn.LookupTableMaskZero
    (2): nn.Linear(128 -> 512)
    (3): nn.LSTM(512 -> 512)
    (4): nn.NormStabilizer
    (5): nn.Linear(512 -> 128)
  }
  (2): nn.CAddTable
  (3): nn.MulConstant
}
####################################################################
----BiLangModelLSTMScAvg----
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
        |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
        |      (1): nn.Linear(128 -> 512)
        |      (2): nn.LSTM(512 -> 512)
        |      (3): nn.NormStabilizer
        |      (4): nn.Dropout(0.5, busy)
        |      (5): nn.Linear(512 -> 128)
        |    }
         `-> (2): nn.Identity
         ... -> output
    }
  }
  (2): nn.FlattenTable
  (3): nn.CAddTable
  (4): nn.MulConstant
}
####################################################################
----BiLangModelLSTM2LyScAvg----
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nn.Sequencer @ nn.Recursor @ nn.LookupTableMaskZero
  (2): nn.Sequencer @ nn.Recursor @ nn.Sequential {
    [input -> (1) -> (2) -> output]
    (1): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Identity
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.ParallelTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
          |      (1): nn.Linear(128 -> 512)
          |      (2): nn.LSTM(512 -> 512)
          |      (3): nn.NormStabilizer
          |      (4): nn.Dropout(0.5, busy)
          |      (5): nn.Linear(512 -> 128)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (3): nn.ConcatTable {
        input
          |`-> (1): nn.Identity
           `-> (2): nn.Identity
           ... -> output
      }
      (4): nn.ParallelTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> output]
          |      (1): nn.NarrowTable
          |      (2): nn.CAddTable
          |    }
           `-> (2): nn.NarrowTable
           ... -> output
      }
      (5): nn.FlattenTable
    }
    (2): nn.ParallelTable {
      input
        |`-> (1): nn.Sequential {
        |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
        |      (1): nn.Linear(128 -> 512)
        |      (2): nn.LSTM(512 -> 512)
        |      (3): nn.NormStabilizer
        |      (4): nn.Dropout(0.5, busy)
        |      (5): nn.Linear(512 -> 128)
        |    }
         `-> (2): nn.Identity
         ... -> output
    }
  }
  (3): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.FlattenTable
    (2): nn.CAddTable
    (3): nn.MulConstant
  }
}
####################################################################
----BiLangModelLSTM3LyScAvg----
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nn.Sequencer @ nn.Recursor @ nn.LookupTableMaskZero
  (2): nn.Sequencer @ nn.Recursor @ nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Identity
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.ParallelTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
          |      (1): nn.Linear(128 -> 512)
          |      (2): nn.LSTM(512 -> 512)
          |      (3): nn.NormStabilizer
          |      (4): nn.Dropout(0.5, busy)
          |      (5): nn.Linear(512 -> 128)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
      (3): nn.ConcatTable {
        input
          |`-> (1): nn.Identity
           `-> (2): nn.Identity
           ... -> output
      }
      (4): nn.ParallelTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> output]
          |      (1): nn.NarrowTable
          |      (2): nn.CAddTable
          |    }
           `-> (2): nn.NarrowTable
           ... -> output
      }
      (5): nn.FlattenTable
    }
    (2): nn.ParallelTable {
      input
        |`-> (1): nn.Sequential {
        |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
        |      (1): nn.Linear(128 -> 512)
        |      (2): nn.LSTM(512 -> 512)
        |      (3): nn.NormStabilizer
        |      (4): nn.Dropout(0.5, busy)
        |      (5): nn.Linear(512 -> 128)
        |    }
         `-> (2): nn.Identity
         ... -> output
    }
    (3): nn.Sequential {
      [input -> (1) -> (2) -> (3) -> output]
      (1): nn.ConcatTable {
        input
          |`-> (1): nn.Identity
           `-> (2): nn.Identity
           ... -> output
      }
      (2): nn.ParallelTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> output]
          |      (1): nn.NarrowTable
          |      (2): nn.CAddTable
          |    }
           `-> (2): nn.NarrowTable
           ... -> output
      }
      (3): nn.ParallelTable {
        input
          |`-> (1): nn.Sequential {
          |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
          |      (1): nn.Linear(128 -> 512)
          |      (2): nn.LSTM(512 -> 512)
          |      (3): nn.NormStabilizer
          |      (4): nn.Dropout(0.5, busy)
          |      (5): nn.Linear(512 -> 128)
          |    }
           `-> (2): nn.Identity
           ... -> output
      }
    }
  }
  (3): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.FlattenTable
    (2): nn.CAddTable
    (3): nn.MulConstant
  }
}
----BiLangModelBiLSTMAvg2----
####################################################################
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
  (1): nn.Sequencer @ nn.Recursor @ nn.LookupTableMaskZero
  (2): nn.ConcatTable {
    input
      |`-> (1): nn.Sequencer @ nn.Recursor @ nn.Sequential {
      |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |      (1): nn.Linear(128 -> 512)
      |      (2): nn.LSTM(512 -> 512)
      |      (3): nn.NormStabilizer
      |      (4): nn.Linear(512 -> 128)
      |    }
       `-> (2): nn.Sequential {
             [input -> (1) -> (2) -> (3) -> output]
             (1): nn.ReverseTable
             (2): nn.Sequencer @ nn.Recursor @ nn.Sequential {
               [input -> (1) -> (2) -> (3) -> (4) -> output]
               (1): nn.Linear(128 -> 512)
               (2): nn.LSTM(512 -> 512)
               (3): nn.NormStabilizer
               (4): nn.Linear(512 -> 128)
             }
             (3): nn.ReverseTable
           }
       ... -> output
  }
  (3): nn.ZipTable
  (4): nn.Sequencer @ nn.Recursor @ nn.JoinTable
  (5): nn.CAddTable
  (6): nn.MulConstant
}
####################################################################
----BiLangModelBiLSTMScAvg2----
Debugging session completed (traced 0 instructions).
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
  (1): nn.Sequencer @ nn.Recursor @ nn.LookupTableMaskZero
  (2): nn.ConcatTable {
    input
      |`-> (1): nn.Sequencer @ nn.Recursor @ nn.Sequential {
      |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
      |      (1): nn.ConcatTable {
      |        input
      |          |`-> (1): nn.Identity
      |           `-> (2): nn.Identity
      |           ... -> output
      |      }
      |      (2): nn.ParallelTable {
      |        input
      |          |`-> (1): nn.Sequential {
      |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |          |      (1): nn.Linear(128 -> 512)
      |          |      (2): nn.LSTM(512 -> 512)
      |          |      (3): nn.NormStabilizer
      |          |      (4): nn.Linear(512 -> 128)
      |          |    }
      |           `-> (2): nn.Identity
      |           ... -> output
      |      }
      |      (3): nn.FlattenTable
      |      (4): nn.CAddTable
      |      (5): nn.MulConstant
      |    }
       `-> (2): nn.Sequential {
             [input -> (1) -> (2) -> (3) -> output]
             (1): nn.ReverseTable
             (2): nn.Sequencer @ nn.Recursor @ nn.Sequential {
               [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
               (1): nn.ConcatTable {
                 input
                   |`-> (1): nn.Identity
                    `-> (2): nn.Identity
                    ... -> output
               }
               (2): nn.ParallelTable {
                 input
                   |`-> (1): nn.Sequential {
                   |      [input -> (1) -> (2) -> (3) -> (4) -> output]
                   |      (1): nn.Linear(128 -> 512)
                   |      (2): nn.LSTM(512 -> 512)
                   |      (3): nn.NormStabilizer
                   |      (4): nn.Linear(512 -> 128)
                   |    }
                    `-> (2): nn.Identity
                    ... -> output
               }
               (3): nn.FlattenTable
               (4): nn.CAddTable
               (5): nn.MulConstant
             }
             (3): nn.ReverseTable
           }
       ... -> output
  }
  (3): nn.ZipTable
  (4): nn.Sequencer @ nn.Recursor @ nn.JoinTable
  (5): nn.CAddTable
  (6): nn.MulConstant
}
####################################################################
