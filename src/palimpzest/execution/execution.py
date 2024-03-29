from palimpzest.sets import Set
from palimpzest.policy import Policy, MaxQuality, UserChoice
from palimpzest.datamanager import DataDirectory

class Execution:
    """An Execution is responsible for completing a query on a given Set.
    Right now we assume the query is always SELECT * FROM set."""
    def __init__(self, rootset: Set, policy: Policy) -> None:
        self.rootset = rootset
        self.policy = policy

    def executeAndOptimize(self):
        """An execution of the rootset, subject to user-given policy."""
        logicalTree = self.rootset.getLogicalTree()

        # Crude algorithm:
        # 1. Generate a logical plan
        # 2. If there's no previously-created reliable data, then get some via sampling on a high-quality plan
        # 3. Once the sample is available, either (1) declare victory because the work is done, or (2) 
        #    use the sample to reoptimize according to user preferences

        cache = DataDirectory().getCacheService()
        cachedInfo = cache.getCachedData("querySamples", self.rootset.universalIdentifier())
        if cachedInfo is not None:
            sampleOutputs, profileData = cachedInfo
        else:
            # Obtain a sample of the first 'sampleSize' records.
            # We need the output examples as well as the performance data.
            sampleSize = 4
            limitSet = self.rootset.limit(sampleSize)
            logicalTree = limitSet.getLogicalTree()
            candidatePlans = logicalTree.createPhysicalPlanCandidates(shouldProfile=True)
            planTime, planCost, quality, physicalTree = MaxQuality().choose(candidatePlans)

            # Execute the physical plan and cache the results
            sampleOutputs = [r for r in physicalTree]
            profileData = physicalTree.getProfilingData()

            # REMIND michjc: now we must transform the profileData object into something that optimizers can use

            # We put this into an ephemeral cache 
            cache.putCachedData("querySamples", self.rootset.universalIdentifier(), (sampleOutputs, profileData))
                                
        # Ok now reoptimize the logical plan, this time with the sample data.
        # (The data is not currently being used; let's see if this method can work first)
        logicalTree = self.rootset.getLogicalTree()
        candidatePlans = logicalTree.createPhysicalPlanCandidates()
        if type(self.policy) == UserChoice:
            def emitNestedTuple(node, indent=0):
                elt, child = node
                print(" " * indent, elt)
                if child is not None:
                    emitNestedTuple(child, indent=indent+2)

            print("----------")
            for idx, cp in enumerate(candidatePlans):
                print(f"Plan {idx}: Time est: {cp[0]:.3f} -- Cost est: {cp[1]:.3f} -- Quality est: {cp[2]:.3f}")
                print("Physical operator tree")
                physicalOps = cp[3].dumpPhysicalTree()
                emitNestedTuple(physicalOps)
                print("----------")
        planTime, planCost, quality, physicalTree = self.policy.choose(candidatePlans)
        return physicalTree

