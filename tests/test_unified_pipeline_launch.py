import unittest
from copy import deepcopy
from length_budget_distill.unified_pipeline_launch import nodes,validate_topology

class UnifiedPipelineTests(unittest.TestCase):
    def test_full_grid_and_all_parent_merges_are_required(self):
        graph=nodes(32,4,1);validate_topology(graph,['raw_merge','steered_merge','text_prepare'])
        self.assertEqual(len(graph),69)
        selected={r['key']:r for r in graph}
        self.assertEqual(len(selected['text_merge']['parents']),64)
        self.assertEqual(set(selected['sft_prepare']['parents']),{'raw_merge','steered_merge','text_merge'})
        self.assertEqual(selected['smoke_merge']['parents'],['dap_smoke','tokenskip_smoke'])
        for method,lanes in (('dap',4),('tokenskip',1)):
            for shard in range(32):
                node=selected[f'{method}_{shard:02d}'];self.assertIn('smoke_merge',node['parents'])
                if shard>=lanes:self.assertIn(f'{method}_{shard-lanes:02d}',node['parents'])
                self.assertEqual(node['cohort'],'student_pool')

    def test_dangling_and_repeated_nodes_are_rejected(self):
        graph=nodes(2,1,1)
        for bad in (graph+graph[:1],list(reversed(graph))):
            with self.assertRaises(ValueError):validate_topology(bad,['raw_merge','steered_merge','text_prepare'])
        changed=deepcopy(graph);changed[-1]['parents']=['unknown']
        with self.assertRaises(ValueError):validate_topology(changed,['raw_merge','steered_merge','text_prepare'])

    def test_training_and_unbounded_lanes_are_not_in_this_stage(self):
        for args in ((2,0,1),(2,3,1),(2,1,3)):
            with self.assertRaises(ValueError):nodes(*args)
        graph=nodes(2,1,1);graph[-1]['stage']='train'
        with self.assertRaises(ValueError):validate_topology(graph,['raw_merge','steered_merge','text_prepare'])

if __name__=='__main__':unittest.main()
