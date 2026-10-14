// Research figures transcribed from the public EncFormer manuscript:
// https://arxiv.org/html/2604.09975 (Tables VII, VIII, IX and Sections III-V).
// These are reported or derived from published work, NOT values measured by this site.
window.ENCFORMER_DATA = {
  networks: {
    LAN: {mbps: 1000, rtt: 0.3},
    WAN1: {mbps: 400, rtt: 4},
    WAN2: {mbps: 100, rtt: 4},
    WAN3: {mbps: 100, rtt: 80},
  },
  benchmarks: {
    "gpt2-base": {
      label: "GPT2-base", tokens: 64,
      BOLT: {LAN: 7.6, WAN1: 12.4, WAN2: 31.8, WAN3: 51.6, comm: 34.8},
      BumbleBee: {LAN: 2.9, WAN1: 3.5, WAN2: 4.9, WAN3: 12.0, comm: 2.5},
      BLB: {LAN: 2.0, WAN1: 2.5, WAN2: 3.9, WAN3: 8.1, comm: 1.5},
      EncFormer: {LAN: 1.6, WAN1: 2.0, WAN2: 3.2, WAN3: 6.0, comm: 1.0},
    },
    "bert-base": {
      label: "BERT-base", tokens: 128,
      BOLT: {LAN: 15.5, WAN1: 18.5, WAN2: 77.9, WAN3: 122.7, comm: 63.6},
      BumbleBee: {LAN: 4.3, WAN1: 6.2, WAN2: 11.6, WAN3: 21.3, comm: 5.8},
      BLB: {LAN: 2.5, WAN1: 3.8, WAN2: 6.6, WAN3: 13.2, comm: 3.0},
      EncFormer: {LAN: 2.1, WAN1: 2.9, WAN2: 4.6, WAN3: 8.9, comm: 2.2},
    },
    "bert-large": {
      label: "BERT-large", tokens: 128,
      BOLT: {LAN: 43.8, WAN1: 49.3, WAN2: 208.2, WAN3: 260.6, comm: 158.9},
      BumbleBee: {LAN: 9.7, WAN1: 15.6, WAN2: 30.8, WAN3: 49.2, comm: 15.2},
      BLB: {LAN: 6.6, WAN1: 9.8, WAN2: 16.2, WAN3: 24.9, comm: 7.8},
      EncFormer: {LAN: 5.3, WAN1: 7.4, WAN2: 12.8, WAN3: 20.6, comm: 5.8},
    },
  },
  accuracy: [
    {name: "SST-2", full: 92.43, teacher: 92.13, ours: 91.78},
    {name: "MRPC (acc)", full: 87.75, teacher: 87.01, ours: 86.76},
    {name: "RTE", full: 72.56, teacher: 69.43, ours: 70.03},
  ],
  sampleVisits: {
    total: 1248, today: 47, countries: [
      {country: "SG", views: 383}, {country: "US", views: 274},
      {country: "GB", views: 151}, {country: "DE", views: 116},
      {country: "IN", views: 108}, {country: "JP", views: 92},
      {country: "AU", views: 77}, {country: "FR", views: 47},
    ],
    daily: [43, 55, 39, 64, 50, 78, 83, 51, 48, 76, 94, 62, 71, 47],
  },
};
